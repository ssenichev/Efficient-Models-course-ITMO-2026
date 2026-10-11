import json
from pathlib import Path
import sys

import pytest
import yaml

from xsa_repro.config import training_plan
from xsa_repro.prepare_data import prepare
from xsa_repro.sweep import (Queue, completed, jobs_for_plan, last_metrics,
                             render_report, resume_command)
from xsa_repro.train import train_run


def tiny_plan(smoke_config, tmp_path):
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(smoke_config))
    return {"deadline": "2000-01-01T12:00:00+03:00", "minimum_free_gib": 0,
            "command_timeout_hours": 1, "poll_seconds": 0.05,
            "workspace": str(tmp_path / "queue"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "groups": [{"config": str(path), "seeds": [42]}]}


def test_scientific_sweep_budgets_and_pairing():
    plan = json.loads(Path("configs/mac_sweep.json").read_text())
    jobs = jobs_for_plan(plan)
    assert len(jobs) == 18
    assert sum(job["target_tokens"] for job in jobs) == 840155136
    for first, second in zip(jobs[::2], jobs[1::2]):
        assert {first["attention"], second["attention"]} == {"sa", "xsa"}
        assert first["config"] == second["config"]
        assert first["config"]["train"]["device"] == "mps"
        assert first["config"]["train"]["precision"] == "fp32"
        assert first["config"]["train"]["micro_batch_size"] == 1
    main = next(j for j in jobs if "100m" in j["config"]["experiment"])
    contexts = [j for j in jobs if "context" in j["config"]["experiment"]]
    assert {j["config"]["model"]["block_size"] for j in contexts} == {256, 512, 1024}
    for job in contexts:
        assert job["config"]["data"] == main["config"]["data"]
        assert job["config"]["train"]["global_batch_tokens"] == 65536
        assert job["target_tokens"] == 30015488
    assert main["target_tokens"] == 100007936
    assert training_plan(main["config"])["grad_accum_steps"] == 64


def test_duplicate_queue_rejected(smoke_config, tmp_path):
    plan = tiny_plan(smoke_config, tmp_path)
    plan["groups"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        jobs_for_plan(plan)


def test_no_partial_metrics_as_final(smoke_config, tmp_path):
    plan = tiny_plan(smoke_config, tmp_path)
    jobs = jobs_for_plan(plan)
    run = Path(jobs[0]["run_dir"])
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text('{"tokens_seen":64,"val_loss":0.123456}\n{"unfinished":')
    assert last_metrics(run)["tokens_seen"] == 64
    state = {"deadline": plan["deadline"], "status": "running",
             "jobs": {str(run): {"status": "training"}}, "verified_pairs": {}}
    text = render_report(state, jobs)
    assert "64 / 256 | — | —" in text
    assert "0.123456" not in text
    assert not completed(jobs[0], "unused")
    with pytest.raises(RuntimeError, match="resumable"):
        resume_command(jobs[0])


def test_queue_offline_resume_deadline_and_retention(smoke_config, tmp_path):
    plan = tiny_plan(smoke_config, tmp_path)
    prepare(smoke_config)
    # Simulate a previous interruption, keeping the original full LR schedule.
    run = train_run(smoke_config, "sa", max_steps=2)
    job = jobs_for_plan(plan)[0]
    assert resume_command(job)[-2:] == ["--resume", str(run / "latest.pt")]
    queue = Queue(plan, tmp_path / "status.json", tmp_path / "report.md", tmp_path / "deadline.md")
    queue.run()
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == "complete"
    assert len(state["verified_pairs"]) == 1
    assert state["deadline_written"]
    assert "Снимок состояния очереди" in (tmp_path / "deadline.md").read_text()
    assert "complete" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "queue/source_snapshot/provenance.json").is_file()
    assert (tmp_path / "artifacts/synthetic_smoke_results.json").is_file()
    for job in queue.jobs:
        assert completed(job, queue.source["source_sha256"])
        path = Path(job["run_dir"])
        for name in ("initial.pt", "latest.pt", "best_val.pt", "final.pt", "retention.json"):
            assert (path / name).stat().st_size > 0
        assert not list(path.glob("step*.pt"))
        assert json.loads((path / "retention.json").read_text())["final_sha256"]
    # Restart is idempotent and checks provenance before reusing any run.
    restarted = Queue(plan, tmp_path / "status.json", tmp_path / "report.md", tmp_path / "deadline.md")
    restarted.run()
    assert restarted.state["status"] == "complete"
    changed = dict(plan, minimum_free_gib=1)
    with pytest.raises(ValueError, match="plan/source changed"):
        Queue(changed, tmp_path / "status.json", tmp_path / "report.md", tmp_path / "deadline.md")


def test_command_failure_recorded_and_disk_preflight(smoke_config, tmp_path):
    plan = tiny_plan(smoke_config, tmp_path)
    queue = Queue(plan, tmp_path / "status.json", tmp_path / "report.md", tmp_path / "deadline.md")
    with pytest.raises(RuntimeError, match="exit 7"):
        queue.command([sys.executable, "-c", "raise SystemExit(7)"], "intentional_failure")
    assert queue.state["last_returncode"] == 7
    assert queue.state["child_pid"] is None
    queue.plan["minimum_free_gib"] = 100000
    with pytest.raises(RuntimeError, match="Insufficient free disk"):
        queue.command([sys.executable, "-c", "raise AssertionError('must not run')"], "forbidden")
    assert not (tmp_path / "queue/logs/forbidden.log").exists()


def test_queue_without_deadline(smoke_config, tmp_path):
    plan = tiny_plan(smoke_config, tmp_path)
    plan.pop("deadline")
    queue = Queue(plan, tmp_path / "status.json", tmp_path / "report.md", tmp_path / "snapshot.md")
    queue.publish()
    assert queue.deadline is None
    assert not queue.state["deadline_written"]
    assert not (tmp_path / "snapshot.md").exists()
    text = (tmp_path / "report.md").read_text()
    assert "Apple M5" not in text
    assert "12:00" not in text
    assert "Final NLL" in text
