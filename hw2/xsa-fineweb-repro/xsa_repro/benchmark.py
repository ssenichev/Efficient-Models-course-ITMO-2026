"""Short synthetic hardware benchmark; never consumes FineWeb or writes training runs."""
import argparse
import gc
import json
import math
import time

import torch

from .config import load_config, training_plan
from .model import GPT, ModelConfig
from .train import make_optimizer
from .utils import (autocast_context, device_memory, hardware_name, resolve_device_precision,
                    seed_all, synchronize_device, versions, write_json)


def benchmark(config, attention_type, warmup_steps=2, steps=5):
    if steps <= 0 or warmup_steps < 0:
        raise ValueError("steps > 0 and warmup_steps >= 0 required")
    t, m = config["train"], config["model"]
    plan = training_plan(config)
    device, precision = resolve_device_precision(t)
    if t["compile"]:
        raise ValueError("This benchmark measures eager execution; set compile=false")
    torch.set_num_threads(t["cpu_threads"])
    seed_all(config["seed"], t["deterministic"])
    model = GPT(ModelConfig(**m), attention_type).to(device).train()
    optimizer = make_optimizer(model, t)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 1000)
    # A fixed synthetic global batch matches model/vocabulary/context and accumulation.
    batches = [torch.randint(m["vocab_size"], (t["micro_batch_size"], m["block_size"] + 1), generator=generator)
               for _ in range(plan["grad_accum_steps"])]
    elapsed, peak_allocated, peak_driver = [], 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup_steps + steps):
        synchronize_device(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for tokens in batches:
            x, y = tokens[:, :-1].to(device), tokens[:, 1:].to(device)
            with autocast_context(device, precision):
                _, loss = model(x, y)
            scaler.scale(loss / plan["grad_accum_steps"]).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite benchmark gradient")
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale:
            raise FloatingPointError("Benchmark update skipped by scaler")
        synchronize_device(device)
        seconds = time.perf_counter() - start
        if step >= warmup_steps:
            elapsed.append(seconds)
        memory = device_memory(device)
        peak_allocated = max(peak_allocated, memory["allocated_bytes"])
        peak_driver = max(peak_driver, memory["driver_bytes"] or 0)
        print(json.dumps({"attention": attention_type, "step": step + 1, "warmup": step < warmup_steps,
                          "seconds": seconds}), flush=True)
    tokens_s = steps * t["global_batch_tokens"] / sum(elapsed)
    result = {"attention": attention_type, "hardware": hardware_name(device), "device": str(device),
              "precision": precision, "params": model.parameter_count(), "context": m["block_size"],
              "global_batch_tokens": t["global_batch_tokens"], "micro_batch_size": t["micro_batch_size"],
              "grad_accum_steps": plan["grad_accum_steps"], "measured_steps": steps, "warmup_steps": warmup_steps,
              "step_seconds": elapsed, "tokens_s": tokens_s,
              "peak_device_bytes": peak_allocated, "peak_mps_driver_bytes": peak_driver,
              "device_memory_measurement": memory["measurement"], "versions": versions(),
              "synthetic": True, "interpretation": "Compute-only estimate; excludes data preparation, evaluation, checkpoints and sustained thermal effects.",
              "estimated_hours_one_model": {str(budget): math.ceil(budget / t["global_batch_tokens"]) * t["global_batch_tokens"] / tokens_s / 3600
                                            for budget in (1000000, 10000000, 100000000)}}
    del model, optimizer, scaler, batches, x, y, loss, norm
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--attention", choices=["sa", "xsa", "both"], default="both")
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out", default="reports/hardware_benchmark.json")
    args = p.parse_args()
    config = load_config(args.config)
    methods = ("sa", "xsa") if args.attention == "both" else (args.attention,)
    results = [benchmark(config, kind, args.warmup_steps, args.steps) for kind in methods]
    write_json(args.out, {"results": results})
    print(json.dumps({"out": args.out, "results": results}), flush=True)


if __name__ == "__main__":
    main()
