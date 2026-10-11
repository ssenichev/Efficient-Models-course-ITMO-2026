"""Single-device controlled paired training with exact local resume."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import time

import torch

from .config import load_config, training_plan, validate
from .data import TokenStream, WindowSampler, verify_manifest
from .diagnostics import diagnose_model
from .evaluate import evaluate_model
from .model import GPT, ModelConfig
from .utils import (autocast_context, canonical_hash, code_provenance, resolve_device_precision,
                    restore_rng, rng_state, seed_all, state_hash, utc_now, versions, write_json,
                    synchronize_device, device_memory, hardware_name)


class WarmupCosine:
    """One-based optimizer step; final update uses min_lr exactly."""
    def __init__(self, optimizer, total_steps, warmup_steps, max_lr, min_lr_ratio):
        self.optimizer = optimizer
        self.total_steps, self.warmup = total_steps, min(warmup_steps, max(0, total_steps - 1))
        self.max_lr, self.min_lr = max_lr, max_lr * min_lr_ratio
        self.step_number = 0

    def set_step(self, step):
        self.step_number = step
        if self.warmup and step <= self.warmup:
            lr = self.max_lr * step / self.warmup
        else:
            progress = (step - self.warmup - 1) / max(1, self.total_steps - self.warmup - 1)
            if step == self.total_steps:
                progress = 1.0
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self):
        return {"step_number": self.step_number, "total_steps": self.total_steps, "warmup": self.warmup,
                "max_lr": self.max_lr, "min_lr": self.min_lr}

    def load_state_dict(self, state):
        for k in ("total_steps", "warmup", "max_lr", "min_lr"):
            if state[k] != getattr(self, k):
                raise ValueError("Scheduler configuration changed")
        self.set_step(state["step_number"])


def make_optimizer(model, train):
    # Decay matrix parameters (including tied embedding once), not LN/bias vectors.
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": train["weight_decay"]},
              {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=train["learning_rate"], betas=tuple(train["betas"]))


def save_checkpoint(path, model, optimizer, scheduler, scaler, sampler, progress, meta):
    state = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "sampler": sampler.state_dict(),
             "rng": rng_state(), "config_sha": meta["config_sha"], "dataset_sha": meta["dataset_sha"],
             "attention": meta["attention"], "resolved_precision": meta["resolved_precision"],
             "resolved_device": meta["resolved_device"], **progress}
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, sampler, meta):
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("config_sha", "dataset_sha", "attention", "resolved_precision", "resolved_device"):
        if state[key] != meta[key]:
            raise ValueError(f"Resume mismatch: {key}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    sampler.load_state_dict(state["sampler"])
    restore_rng(state["rng"])
    return {k: state[k] for k in ("step", "tokens_seen", "best_val", "best_checkpoint", "training_seconds", "wall_seconds",
                                 "peak_vram_bytes", "peak_device_bytes", "peak_mps_driver_bytes",
                                 "last_train_loss", "benchmark_seconds", "benchmark_tokens")}


LOG_FIELDS = ["step", "tokens_seen", "learning_rate", "train_loss", "val_loss", "val_ppl", "evaluated_tokens",
              "throughput_tokens_per_sec", "peak_vram_bytes", "peak_device_bytes", "peak_mps_driver_bytes",
              "training_seconds", "wall_seconds", "full_val"]


def append_log(run_dir, row):
    row = {key: row.get(key) for key in LOG_FIELDS}
    with (run_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    path = run_dir / "metrics.csv"
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row))


def truncate_logs(run_dir, step):
    path = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    rows = [row for row in rows if row["step"] <= step]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def train_run(config, attention_type, seed=None, resume=None, max_steps=None):
    config = copy.deepcopy(config)
    if seed is not None:
        config["seed"] = seed
    validate(config)
    t, m = config["train"], config["model"]
    plan = training_plan(config)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    run_dir = Path(config["runs_dir"]) / config["experiment"] / f'{attention_type}_seed{config["seed"]}'
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run exists: {run_dir}; choose --resume or a new experiment")
    if resume and not run_dir.exists():
        raise FileNotFoundError("Resume requires the existing run directory")
    data_dir = Path(config["data"]["processed_dir"]).resolve()
    manifest, dataset_sha = verify_manifest(data_dir)
    if manifest["synthetic"] != (config["data"]["source"] == "synthetic"):
        raise ValueError("Synthetic/FineWeb corpus mismatch")
    expected_data = {k: v for k, v in config["data"].items() if k != "processed_dir"}
    actual_data = {k: v for k, v in manifest["data_config"].items() if k != "processed_dir"}
    if expected_data != actual_data:
        raise ValueError("Prepared corpus data configuration differs from experiment config")
    if manifest["tokenizer"]["vocab_size"] != m["vocab_size"] or manifest["tokenizer"]["eot_id"] != config["data"]["eos_id"]:
        raise ValueError("Corpus tokenizer/vocabulary differs from model")
    if not manifest["synthetic"] and config["data"]["revision"] and config["data"]["revision"] != manifest["revision"]:
        raise ValueError("Corpus revision differs from experiment config")
    device, precision = resolve_device_precision(t)
    if device.type == "mps" and t["compile"]:
        raise ValueError("MPS experiments currently require compile=false")
    torch.set_num_threads(t["cpu_threads"])
    seed_all(config["seed"], t["deterministic"])
    model = GPT(ModelConfig(**m), attention_type).to(device)
    with torch.device("meta"):
        other = GPT(ModelConfig(**m), "xsa" if attention_type == "sa" else "sa")
    if other.parameter_count() != model.parameter_count():
        raise AssertionError("SA and XSA parameter counts differ")
    train_stream = TokenStream(data_dir / "train.bin", m["block_size"], m["vocab_size"])
    val_stream = TokenStream(data_dir / "val.bin", m["block_size"], m["vocab_size"])
    sampler = WindowSampler(train_stream, config["seed"])
    optimizer = make_optimizer(model, t)
    scheduler = WarmupCosine(optimizer, plan["total_steps"], t["warmup_steps"], t["learning_rate"], t["min_lr_ratio"])
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    meta = {"config": config, "config_sha": canonical_hash(config), "dataset_sha": dataset_sha,
            "attention": attention_type, "seed": config["seed"], "params": model.parameter_count(),
            "initial_state_sha256": state_hash(model.state_dict()), "data_dir": str(data_dir),
            "resolved_device": str(device), "resolved_precision": precision, "plan": plan,
            "versions": versions(), "code": code_provenance(), "created_utc": utc_now(),
            "hardware": hardware_name(device), "device_memory_measurement": device_memory(device)["measurement"],
            "synthetic": manifest["synthetic"], "sampler": "PCG64 uniform with replacement; starts multiple of T",
            "eval_window_starts": val_stream.validation_starts(t["eval_max_windows"]).tolist()}
    progress = {"step": 0, "tokens_seen": 0, "best_val": None, "best_checkpoint": None, "training_seconds": 0.0, "wall_seconds": 0.0,
                "peak_vram_bytes": 0, "peak_device_bytes": 0, "peak_mps_driver_bytes": 0,
                "last_train_loss": None, "benchmark_seconds": 0.0, "benchmark_tokens": 0}
    if resume:
        previous = json.loads((run_dir / "run_manifest.json").read_text())
        for field in ("config_sha", "dataset_sha", "initial_state_sha256", "resolved_precision", "resolved_device"):
            if previous[field] != meta[field]:
                raise ValueError(f"Run configuration changed: {field}")
        if previous["code"]["source_sha256"] != meta["code"]["source_sha256"]:
            raise ValueError("Source changed; resume with the exact saved source version")
        meta = previous
        progress = load_checkpoint(resume, model, optimizer, scheduler, scaler, sampler, meta)
        if progress["tokens_seen"] != progress["step"] * t["global_batch_tokens"]:
            raise ValueError("Checkpoint token count differs from step count")
        truncate_logs(run_dir, progress["step"])
        # Retained best versions make rewinding safe, including an initial-state resume.
        if progress["best_checkpoint"]:
            import shutil
            shutil.copyfile(run_dir / progress["best_checkpoint"], run_dir / "best_val.pt")
        if progress["step"] < plan["total_steps"]:
            for name in ("final.pt", "summary.json", "evaluation_final.json", "diagnostics.json"):
                artifact = run_dir / name
                if artifact.exists():
                    artifact.replace(artifact.with_suffix(artifact.suffix + ".stale"))
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir / "run_manifest.json", meta)
        import yaml
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        save_checkpoint(run_dir / "initial.pt", model, optimizer, scheduler, scaler, sampler, progress, meta)
        metrics = evaluate_model(model, val_stream, t["eval_batch_size"], device, precision, t["eval_max_windows"])
        progress["best_val"] = metrics["val_loss"]
        progress["best_checkpoint"] = "best_val_step000000.pt"
        save_checkpoint(run_dir / progress["best_checkpoint"], model, optimizer, scheduler, scaler, sampler, progress, meta)
        save_checkpoint(run_dir / "best_val.pt", model, optimizer, scheduler, scaler, sampler, progress, meta)
        append_log(run_dir, {**progress, **metrics, "learning_rate": 0.0, "full_val": False})
    print(json.dumps({"event": "startup", "attention": attention_type, "params": meta["params"],
                      "device": str(device), "precision": precision, **plan, "initial_state_sha256": meta["initial_state_sha256"]}))
    runner = torch.compile(model) if t["compile"] else model
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    target_step = plan["total_steps"] if max_steps is None else min(max_steps, plan["total_steps"])
    if target_step < progress["step"]:
        raise ValueError("max_steps precedes resume step")
    base_wall, wall_start = progress["wall_seconds"], time.perf_counter()
    if progress["best_val"] is None:
        # initial.pt is saved before the initial validation, and is a valid resume point.
        metrics = evaluate_model(model, val_stream, t["eval_batch_size"], device, precision, t["eval_max_windows"])
        progress["best_val"] = metrics["val_loss"]
        progress["best_checkpoint"] = "best_val_step000000.pt"
        save_checkpoint(run_dir / progress["best_checkpoint"], model, optimizer, scheduler, scaler, sampler, progress, meta)
        save_checkpoint(run_dir / "best_val.pt", model, optimizer, scheduler, scaler, sampler, progress, meta)
    for step in range(progress["step"] + 1, target_step + 1):
        model.train()
        synchronize_device(device)
        start = time.perf_counter()
        lr = scheduler.set_step(step)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(plan["grad_accum_steps"]):
            x, y = train_stream.batch(sampler.sample(t["micro_batch_size"]), device)
            with autocast_context(device, precision):
                _, loss = runner(x, y)
            loss_sum += loss.detach().float().item() / plan["grad_accum_steps"]
            scaler.scale(loss / plan["grad_accum_steps"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
        if not math.isfinite(loss_sum) or not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite loss/gradient: no optimizer step counted; resume from latest checkpoint")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise FloatingPointError("FP16 optimizer update skipped; no tokens counted; resume with a stable shared configuration")
        synchronize_device(device)
        seconds = time.perf_counter() - start
        progress.update(step=step, tokens_seen=step * t["global_batch_tokens"], last_train_loss=loss_sum,
                        training_seconds=progress["training_seconds"] + seconds)
        if step > min(5, plan["total_steps"] // 10):
            progress["benchmark_seconds"] += seconds
            progress["benchmark_tokens"] += t["global_batch_tokens"]
        if device.type == "cuda":
            progress["peak_vram_bytes"] = max(progress["peak_vram_bytes"], torch.cuda.max_memory_allocated(device))
        memory = device_memory(device)
        progress["peak_device_bytes"] = max(progress["peak_device_bytes"], memory["allocated_bytes"])
        progress["peak_mps_driver_bytes"] = max(progress["peak_mps_driver_bytes"], memory["driver_bytes"] or 0)
        metrics = {}
        if step % t["eval_every_steps"] == 0 or step == plan["total_steps"]:
            metrics = evaluate_model(model, val_stream, t["eval_batch_size"], device, precision, t["eval_max_windows"])
            if metrics["val_loss"] < progress["best_val"]:
                progress["best_val"] = metrics["val_loss"]
                progress["best_checkpoint"] = f"best_val_step{step:06d}.pt"
                progress["wall_seconds"] = base_wall + time.perf_counter() - wall_start
                save_checkpoint(run_dir / progress["best_checkpoint"], model, optimizer, scheduler, scaler, sampler, progress, meta)
                save_checkpoint(run_dir / "best_val.pt", model, optimizer, scheduler, scaler, sampler, progress, meta)
        progress["wall_seconds"] = base_wall + time.perf_counter() - wall_start
        if step % t["log_every_steps"] == 0 or metrics or step == target_step:
            append_log(run_dir, {**progress, **metrics, "train_loss": loss_sum, "learning_rate": lr,
                                "throughput_tokens_per_sec": t["global_batch_tokens"] / seconds, "full_val": False})
        if step % t["checkpoint_every_steps"] == 0 or step == target_step:
            save_checkpoint(run_dir / "latest.pt", model, optimizer, scheduler, scaler, sampler, progress, meta)
            if step % t["checkpoint_every_steps"] == 0:
                save_checkpoint(run_dir / f"step{step:06d}.pt", model, optimizer, scheduler, scaler, sampler, progress, meta)
    if progress["step"] == plan["total_steps"]:
        final_metrics = evaluate_model(model, val_stream, t["eval_batch_size"], device, precision)
        diagnostics = diagnose_model(model, val_stream, t["eval_batch_size"], device, precision, t["diagnostic_max_windows"])
        diagnostics.update(tokens_seen=progress["tokens_seen"], checkpoint="final.pt", dataset_sha=dataset_sha)
        write_json(run_dir / "diagnostics.json", diagnostics)
        progress["wall_seconds"] = base_wall + time.perf_counter() - wall_start
        if device.type == "cuda":
            progress["peak_vram_bytes"] = max(progress["peak_vram_bytes"], torch.cuda.max_memory_allocated(device))
        synchronize_device(device)
        memory = device_memory(device)
        progress["peak_device_bytes"] = max(progress["peak_device_bytes"], memory["allocated_bytes"])
        progress["peak_mps_driver_bytes"] = max(progress["peak_mps_driver_bytes"], memory["driver_bytes"] or 0)
        save_checkpoint(run_dir / "final.pt", model, optimizer, scheduler, scaler, sampler, progress, meta)
        final_metrics.update(full_val=True, checkpoint="final.pt", tokens_seen=progress["tokens_seen"], step=progress["step"],
                             attention=attention_type, precision=precision, dataset_sha=dataset_sha)
        write_json(run_dir / "evaluation_final.json", final_metrics)
        append_log(run_dir, {**progress, **final_metrics, "train_loss": progress["last_train_loss"],
                            "learning_rate": scheduler.optimizer.param_groups[0]["lr"], "full_val": True})
        summary = {"attention": attention_type, "seed": config["seed"], "tokens_final": progress["tokens_seen"],
                   "train_loss": progress["last_train_loss"], "val_loss": final_metrics["val_loss"],
                   "ppl": final_metrics["val_ppl"], "evaluated_tokens": final_metrics["evaluated_tokens"],
                   "seconds": progress["wall_seconds"], "training_seconds": progress["training_seconds"],
                   "tokens_s": progress["tokens_seen"] / max(progress["training_seconds"], 1e-12),
                   "steady_tokens_s": progress["benchmark_tokens"] / max(progress["benchmark_seconds"], 1e-12),
                   "peak_gpu_bytes": progress["peak_vram_bytes"], "params": meta["params"], "dataset_sha": dataset_sha,
                   "peak_device_bytes": progress["peak_device_bytes"], "peak_mps_driver_bytes": progress["peak_mps_driver_bytes"],
                   "device_memory_measurement": meta["device_memory_measurement"], "device": str(device),
                   "config_sha": meta["config_sha"], "initial_state_sha256": meta["initial_state_sha256"],
                   "synthetic": meta["synthetic"], "precision": precision, "projection_fp32": m["xsa_projection_fp32"],
                   "block_size": m["block_size"], "best_subset_val_loss": progress["best_val"]}
        write_json(run_dir / "summary.json", summary)
    return run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--attention", required=True, choices=["sa", "xsa"])
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", help="Path to a trusted local initial/latest/step checkpoint")
    p.add_argument("--max-steps", type=int, help="Stop early for debugging; retain original scheduler/budget")
    p.add_argument("--micro-batch-size", type=int)
    p.add_argument("--device")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    args = p.parse_args()
    config = load_config(args.config)
    for field in ("micro_batch_size", "device", "precision", "compile"):
        value = getattr(args, field)
        if value is not None:
            config["train"][field] = value
    train_run(config, args.attention, args.seed, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
