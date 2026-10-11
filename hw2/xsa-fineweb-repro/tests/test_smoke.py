import json
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from xsa_repro.data import TokenStream
from xsa_repro.diagnostics import CosineAccumulator, diagnose_run
from xsa_repro.evaluate import evaluate_model, evaluate_run, load_run
from xsa_repro.plot import check_pairs, collect_run, plot_runs
from xsa_repro.prepare_data import prepare
from xsa_repro.train import train_run


def test_offline_end_to_end(smoke_config, tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Offline smoke attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    prepare(smoke_config)
    dirs = [train_run(smoke_config, kind) for kind in ("sa", "xsa")]
    for run in dirs:
        for filename in ("initial.pt", "latest.pt", "best_val.pt", "final.pt", "metrics.csv", "metrics.jsonl"):
            assert (run / filename).stat().st_size > 0
        metrics = evaluate_run(run, full_val=True)
        assert metrics["evaluated_tokens"] > 0 and np.isfinite(metrics["val_loss"])
        diag = diagnose_run(run)
        assert diag["windows"] == 8
        for layer in diag["layers"]:
            assert layer["pre"]["valid"] > 0 and layer["pre"]["excluded_first"] == 32
            if diag["attention"] == "xsa":
                assert layer["post"]["mean_abs"] < 2e-5
    metas = [json.loads((r / "run_manifest.json").read_text()) for r in dirs]
    assert metas[0]["initial_state_sha256"] == metas[1]["initial_state_sha256"]
    out = tmp_path / "figures"
    results = plot_runs(dirs, out)
    assert results["synthetic"] and len(results["paired_comparisons"]) == 1
    for name in ("train_loss_vs_tokens", "val_loss_vs_tokens", "perplexity_vs_tokens", "cosine_similarity_by_layer"):
        path = out / f"{name}.png"
        assert path.stat().st_size > 5000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "Synthetic software smoke" in (out / "report_draft.md").read_text()
    bad = [collect_run(path) for path in dirs]
    bad[1]["summary"]["tokens_final"] += 64
    with pytest.raises(ValueError, match="Unfair"):
        check_pairs(bad)


def test_validation_is_token_weighted(smoke_config):
    prepare(smoke_config)
    run = train_run(smoke_config, "sa")
    model, stream, meta, state, device, precision = load_run(run)
    # Three windows with batches 2+1 detect incorrect averaging of batch means.
    result = evaluate_model(model, stream, 2, device, max_windows=3)
    model.eval()
    with torch.no_grad():
        x, y = stream.batch([0, 16, 32])
        logits, _ = model(x)
        ref = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
    assert result["evaluated_tokens"] == 48
    assert result["val_loss"] == pytest.approx(ref, abs=1e-6)


def test_cosine_exclusions():
    a = torch.ones(1, 2, 4, 4)
    v = a.clone()
    a[:, :, 1] = 0
    v[:, :, 2] *= 1e-12
    stats = CosineAccumulator()
    stats.update(a, v)
    result = stats.result()
    assert result["excluded_first"] == 2 and result["excluded_norm"] == 4
    assert result["valid"] == 2 and result["mean"] == 1


def test_cli_contract():
    for module in ("prepare_data", "train", "evaluate", "diagnostics", "plot"):
        completed = subprocess.run([sys.executable, "-m", f"xsa_repro.{module}", "--help"], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
