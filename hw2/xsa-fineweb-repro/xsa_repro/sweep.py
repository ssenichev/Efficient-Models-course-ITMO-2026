"""Durable serial experiment queue, verified pairs and deadline snapshot report.

No shell commands or external messages. Stop with a local STOP file; a running
command is allowed to finish so its checkpoints remain usable. Restart the same
CLI to resume compatible runs. Historical checkpoints are compacted only after
successful final evaluation and pair verification; primary artifacts stay intact.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from .config import load_config, training_plan
from .data import verify_manifest
from .plot import check_pairs, collect_run
from .utils import canonical_hash, code_provenance, file_hash, utc_now, write_json


def jobs_for_plan(plan):
    jobs = []
    seen = set()
    for group in plan["groups"]:
        config = load_config(group["config"])
        for seed in group["seeds"]:
            # Alternate order to reduce a systematic first-method timing bias.
            for attention in (("sa", "xsa") if seed % 2 == 0 else ("xsa", "sa")):
                cfg = copy.deepcopy(config)
                cfg["seed"] = seed
                run = Path(cfg["runs_dir"]) / cfg["experiment"] / f"{attention}_seed{seed}"
                key = str(run)
                if key in seen:
                    raise ValueError(f"Duplicate queued run: {key}")
                seen.add(key)
                jobs.append({"config_path": group["config"], "config": cfg,
                             "attention": attention, "seed": seed, "run_dir": key,
                             "target_tokens": training_plan(cfg)["actual_tokens_budget"]})
    return jobs


def completed(job, source_sha):
    run = Path(job["run_dir"])
    required = ("final.pt", "summary.json", "evaluation_final.json", "diagnostics.json", "run_manifest.json")
    if not all((run / name).is_file() for name in required):
        return False
    entry = collect_run(run)
    meta, summary = entry["meta"], entry["summary"]
    if meta["config_sha"] != canonical_hash(job["config"]) or meta["code"]["source_sha256"] != source_sha:
        raise ValueError(f"Incompatible completed run: {run}")
    if summary["tokens_final"] != job["target_tokens"] or summary["attention"] != job["attention"]:
        raise ValueError(f"Wrong final budget/method: {run}")
    if not all(math.isfinite(summary[k]) for k in ("val_loss", "ppl", "train_loss")):
        raise ValueError(f"Non-finite completed metrics: {run}")
    return True


def resume_command(job):
    command = [sys.executable, "-u", "-m", "xsa_repro.train", "--config", job["config_path"],
               "--attention", job["attention"], "--seed", str(job["seed"])]
    run = Path(job["run_dir"])
    if run.exists():
        for name in ("latest.pt", "initial.pt"):
            if (run / name).exists():
                return command + ["--resume", str(run / name)]
        raise RuntimeError(f"Existing run lacks a resumable checkpoint: {run}")
    return command


def verify_and_compact_pair(dirs):
    entries = [collect_run(run) for run in dirs]
    comparison = check_pairs(entries)
    if len(comparison) != 1:
        raise ValueError("Expected a complete comparable pair")
    states = [torch.load(Path(run) / "final.pt", map_location="cpu", weights_only=False) for run in dirs]
    for key in ("sampler", "scheduler", "tokens_seen", "step"):
        if states[0][key] != states[1][key]:
            raise ValueError(f"Final paired state differs: {key}")
    hashes = {str(run): file_hash(Path(run) / "final.pt") for run in dirs}
    removed = []
    for run, state in zip(map(Path, dirs), states):
        # Keep the best version referenced by both final and latest checkpoints.
        keep = {state["best_checkpoint"]}
        latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        keep.add(latest["best_checkpoint"])
        del latest
        for artifact in [*run.glob("best_val_step*.pt"), *run.glob("step*.pt")]:
            if artifact.name not in keep:
                artifact.unlink()
                removed.append(str(artifact))
        write_json(run / "retention.json", {"verified_utc": utc_now(), "final_sha256": hashes[str(run)],
                   "policy": "Keep initial/latest/best_val/final and referenced best versions; compact completed pairs only",
                   "rewind_limitation": "Historical step checkpoints have been compacted; use initial/latest/final for resume",
                   "removed": [name for name in removed if str(Path(name).parent) == str(run)]})
    return {"comparison": comparison[0], "final_checkpoint_sha256": hashes,
            "final_sampler_equal": True, "final_scheduler_equal": True}


def last_metrics(run):
    path = Path(run) / "metrics.jsonl"
    if not path.exists():
        return None
    # A partially written final line is ignored; never present it as a result.
    for line in reversed(path.read_text().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def render_report(state, jobs, deadline_snapshot=False):
    lines = ["# XSA: результаты серии", "",
             f"Обновлено: {utc_now()}. Статус: **{state['status']}**.", ""]
    if state.get("deadline"):
        lines += [f"Время снимка: {state['deadline']}.", ""]
    if deadline_snapshot:
        lines += ["Снимок состояния очереди. Незавершённые запуски не имеют итоговых метрик.", ""]
    lines += ["## Запуски", "",
              "| Experiment | Method | Seed | T | Global batch tokens | Status | Processed / target tokens | Final NLL | Final PPL |",
              "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for job in jobs:
        name = job["run_dir"]
        record = state["jobs"].get(name, {})
        summary = record.get("summary")
        progress = last_metrics(name)
        tokens = summary["tokens_final"] if summary else (progress or {}).get("tokens_seen", 0)
        nll = f"{summary['val_loss']:.6f}" if summary else "—"
        ppl = f"{summary['ppl']:.4f}" if summary else "—"
        cfg = job["config"]
        lines.append(f"| {cfg['experiment']} | {job['attention']} | {job['seed']} | "
                     f"{cfg['model']['block_size']} | {cfg['train']['global_batch_tokens']} | "
                     f"{record.get('status', 'pending')} | {tokens:,} / {job['target_tokens']:,} | {nll} | {ppl} |")
    lines += ["", "## Проверенные пары", ""]
    if not state["verified_pairs"]:
        lines.append("Завершённых проверенных пар пока нет.")
    groups = {}
    for key, proof in state["verified_pairs"].items():
        p = proof["comparison"]
        lines.append(f"- {key}: ΔNLL(SA−XSA) = **{p['delta_nll_sa_minus_xsa']:.6f}** при {p['tokens_final']:,} tokens. "
                     "Совпадают initial weights, config, corpus, source и итоговые состояния sampler/scheduler.")
        groups.setdefault(p["experiment"], []).append(p["delta_nll_sa_minus_xsa"])
    for name, values in groups.items():
        if len(values) > 1:
            mean = sum(values) / len(values)
            std = math.sqrt(sum((x - mean)**2 for x in values) / (len(values) - 1))
            lines.append(f"- {name}: paired ΔNLL mean ± sample std = {mean:.6f} ± {std:.6f}, n={len(values)} seeds.")
    lines += ["", "## Протокол", "",
              "NLL взвешена по токенам; PPL=exp(NLL). Full validation использует непересекающиеся окна, "
              "короткий хвост исключён. Итоговые checkpoints сравниваются при равных tokens_seen. "
              "Cosine после XSA проверяет проекцию; качество оценивается отдельно по NLL/PPL.", "",
              "После проверки пары сохраняются initial/latest/best_val/final и версии best, "
              "на которые ссылаются final/latest. Прочие periodic/best checkpoints удаляются. "
              "Исходники, план и логи сохраняются в workspace очереди."]
    if state.get("error"):
        lines += ["", "## Ошибка очереди", "", state["error"]]
    for name, item in state["jobs"].items():
        if item.get("error"):
            lines += ["", f"Ошибка {name}: {item['error']}"]
    return "\n".join(lines) + "\n"


class Queue:
    def __init__(self, plan, status_path, report_path, deadline_report_path):
        self.plan = plan
        self.jobs = jobs_for_plan(plan)
        self.status_path, self.report_path, self.deadline_report_path = map(Path, (status_path, report_path, deadline_report_path))
        self.source = code_provenance()
        self.deadline = datetime.fromisoformat(plan["deadline"]) if plan.get("deadline") else None
        if self.deadline is not None and self.deadline.tzinfo is None:
            raise ValueError("Deadline must include a timezone")
        self.state = {"plan_sha": canonical_hash(plan), "source": self.source, "deadline": plan.get("deadline"),
                      "started_utc": utc_now(), "status": "starting", "pid": os.getpid(),
                      "jobs": {}, "verified_pairs": {}, "deadline_written": False}
        if self.status_path.exists():
            previous = json.loads(self.status_path.read_text())
            if previous["plan_sha"] != self.state["plan_sha"] or previous["source"]["source_sha256"] != self.source["source_sha256"]:
                raise ValueError("Queue plan/source changed; use original snapshot or a new status path")
            self.state = previous
            self.state.update(pid=os.getpid(), status="restarting", error=None)
        self.workspace = Path(plan.get("workspace", "runs/sweep"))
        self.artifacts = Path(plan.get("artifacts_dir", "reports"))
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = float(plan.get("poll_seconds", 15))
        if not 0.05 <= self.poll_seconds <= 60:
            raise ValueError("Queue poll_seconds must be between 0.05 and 60")
        (self.workspace / "logs").mkdir(parents=True, exist_ok=True)

    def publish(self):
        self.state["updated_utc"] = utc_now()
        self.state["free_gib"] = shutil.disk_usage(".").free / 2**30
        write_json(self.status_path, self.state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.report_path.with_suffix(".tmp")
        tmp.write_text(render_report(self.state, self.jobs))
        tmp.replace(self.report_path)
        if not self.state["deadline_written"] and self.deadline is not None and datetime.now(timezone.utc) >= self.deadline:
            self.deadline_report_path.parent.mkdir(parents=True, exist_ok=True)
            self.deadline_report_path.write_text(render_report(self.state, self.jobs, True))
            self.state["deadline_written"] = True
            write_json(self.status_path, self.state)

    def archive(self):
        snapshot = self.workspace / "source_snapshot"
        for name, digest in self.source["files"].items():
            target = snapshot / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(name, target)
            if file_hash(target) != digest:
                raise ValueError(f"Source snapshot mismatch: {name}")
        write_json(snapshot / "sweep_plan.json", self.plan)
        write_json(snapshot / "provenance.json", self.source)

    def command(self, command, label):
        if shutil.disk_usage(".").free < self.plan["minimum_free_gib"] * 2**30:
            raise RuntimeError("Insufficient free disk; queue stopped before starting another command")
        log = self.workspace / "logs" / (label + ".log")
        self.state.update(active_command=command, active_log=str(log))
        self.publish()
        print(json.dumps({"event": "command", "command": command, "log": str(log)}), flush=True)
        started = time.monotonic()
        with log.open("a") as output:
            output.write("\nCOMMAND " + json.dumps(command) + "\n")
            output.flush()
            env = os.environ.copy()
            env.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))
            env.setdefault("TIKTOKEN_CACHE_DIR", str(Path("data/tiktoken_cache").resolve()))
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
            self.state["child_pid"] = process.pid
            try:
                while process.poll() is None:
                    if time.monotonic() - started > self.plan["command_timeout_hours"] * 3600:
                        raise TimeoutError(f"Command timeout: {label}")
                    if shutil.disk_usage(".").free < self.plan["minimum_free_gib"] * 2**30:
                        raise RuntimeError("Free disk threshold reached during command")
                    self.publish()
                    time.sleep(self.poll_seconds)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.state["child_pid"] = None
        self.state["last_returncode"] = process.returncode
        self.publish()
        if process.returncode:
            raise RuntimeError(f"Command {label} failed with exit {process.returncode}; see {log}")

    def analyze(self):
        grouped = {}
        for job in self.jobs:
            if self.state["jobs"].get(job["run_dir"], {}).get("status") == "complete":
                grouped.setdefault((job["config"]["experiment"], job["seed"]), {})[job["attention"]] = job["run_dir"]
        complete_dirs = {}
        for (experiment, seed), group in grouped.items():
            if set(group) != {"sa", "xsa"}:
                continue
            dirs = [group["sa"], group["xsa"]]
            key = f"{experiment}/seed{seed}"
            if key not in self.state["verified_pairs"]:
                self.state["verified_pairs"][key] = verify_and_compact_pair(dirs)
            complete_dirs.setdefault(experiment, []).extend(dirs)
        for experiment, dirs in complete_dirs.items():
            output = self.artifacts / "figures/sweep" / experiment
            self.command([sys.executable, "-m", "xsa_repro.plot", "--run-dirs", *dirs,
                          "--out-dir", str(output)], "plot_" + experiment)
            shutil.copyfile(output / "results.json", self.artifacts / (experiment + "_results.json"))
        context_dirs = [run for name, dirs in complete_dirs.items() if "context" in name for run in dirs]
        if context_dirs:
            self.command([sys.executable, "-m", "xsa_repro.plot", "--run-dirs", *context_dirs,
                          "--out-dir", str(self.artifacts / "figures/sweep/context_ablation")], "plot_context_ablation")

    def run(self):
        self.archive()
        self.state["status"] = "running"
        self.publish()
        for job in self.jobs:
            if (self.workspace / "STOP").exists():
                self.state["status"] = "stopped_by_file"
                self.publish()
                return
            if code_provenance()["source_sha256"] != self.source["source_sha256"]:
                raise RuntimeError("Source changed while queue was running; stopping before next command")
            key = job["run_dir"]
            item = self.state["jobs"].setdefault(key, {})
            try:
                if not completed(job, self.source["source_sha256"]):
                    item.update(status="preparing_data", error=None)
                    self.publish()
                    data = Path(job["config"]["data"]["processed_dir"])
                    if not data.exists():
                        for attempt in range(3):
                            try:
                                self.command([sys.executable, "-u", "-m", "xsa_repro.prepare_data", "--config", job["config_path"]],
                                             data.name + "_prepare")
                                break
                            except RuntimeError:
                                if attempt == 2 or data.exists():
                                    raise
                    manifest, _ = verify_manifest(data)
                    expected = {k: v for k, v in job["config"]["data"].items() if k != "processed_dir"}
                    actual = {k: v for k, v in manifest["data_config"].items() if k != "processed_dir"}
                    if expected != actual:
                        raise ValueError("Prepared data does not match queued data configuration")
                    item["status"] = "training"
                    self.publish()
                    label = Path(key).parent.name + "_" + Path(key).name
                    for attempt in range(2):
                        try:
                            self.command(resume_command(job), label)
                            break
                        except RuntimeError:
                            tail = (self.workspace / "logs" / (label + ".log")).read_text()[-20000:]
                            fatal = ("Non-finite", "out of memory", "Source changed", "mismatch", "MPS unavailable")
                            if attempt or any(text in tail for text in fatal) or not (Path(key) / "initial.pt").exists():
                                raise
                    if not completed(job, self.source["source_sha256"]):
                        raise RuntimeError("Training exited without verified full-budget artifacts")
                item.update(status="complete", summary=collect_run(key)["summary"], finished_utc=utc_now())
                self.analyze()
            except (RuntimeError, ValueError, OSError, TimeoutError) as error:
                item.update(status="failed", error=str(error), finished_utc=utc_now())
                # A failed run is never interpreted as a comparable final result.
                print(json.dumps({"event": "failed", "run": key, "error": str(error)}), flush=True)
            self.publish()
        self.state["status"] = "complete" if all(v["status"] == "complete" for v in self.state["jobs"].values()) else "complete_with_failures"
        self.state["finished_utc"] = utc_now()
        self.publish()
        if not self.state["deadline_written"]:
            self.deadline_report_path.write_text(render_report(self.state, self.jobs, True))
            self.state["deadline_written"] = True
            write_json(self.status_path, self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/mac_sweep.json")
    parser.add_argument("--status", default="reports/sweep_status.json")
    parser.add_argument("--report", default="reports/sweep_report.md")
    parser.add_argument("--deadline-report", default="reports/sweep_snapshot.md")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    Path(args.status).parent.mkdir(parents=True, exist_ok=True)
    lock = Path(args.status).with_suffix(".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another queue already holds this status lock")
    queue = Queue(plan, args.status, args.report, args.deadline_report)
    inhibitor = None
    if sys.platform == "darwin":
        inhibitor = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        queue.run()
    except BaseException as error:
        queue.state.update(status="failed", error=str(error))
        queue.publish()
        if not queue.state["deadline_written"]:
            queue.deadline_report_path.write_text(render_report(queue.state, queue.jobs, True))
        raise
    finally:
        if inhibitor is not None:
            inhibitor.terminate()
            inhibitor.wait()
        lock.close()


if __name__ == "__main__":
    main()
