"""Real Metal tests skip on hosts/sandboxes without GPU access."""
import copy
import json
import pytest
import torch

from xsa_repro.diagnostics import diagnose_run
from xsa_repro.evaluate import evaluate_run
from xsa_repro.prepare_data import prepare
from xsa_repro.plot import plot_runs
from xsa_repro.train import train_run
from xsa_repro.utils import rng_state, restore_rng, seed_all

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU not available")


@pytest.mark.parametrize("kind", ["sa", "xsa"])
def test_real_mps_training_evaluation_and_resume(smoke_config, kind):
    smoke_config["train"].update(device="mps", precision="fp32", deterministic=False)
    prepare(smoke_config)
    full_config = copy.deepcopy(smoke_config)
    full_config["experiment"] = "metal_full"
    full_dir = train_run(full_config, kind)
    part_config = copy.deepcopy(smoke_config)
    part_config["experiment"] = "metal_resume"
    part_dir = train_run(part_config, kind, max_steps=2)
    train_run(part_config, kind, resume=part_dir / "latest.pt")
    full = torch.load(full_dir / "final.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(part_dir / "final.pt", map_location="cpu", weights_only=False)
    assert full["resolved_device"] == resumed["resolved_device"] == "mps"
    assert full["sampler"] == resumed["sampler"]
    assert full["tokens_seen"] == resumed["tokens_seen"] == 256
    assert torch.equal(full["rng"]["mps"], resumed["rng"]["mps"])
    for key, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], atol=1e-6, rtol=1e-5)
    metrics = evaluate_run(part_dir, full_val=True)
    assert metrics["evaluated_tokens"] == 1120
    diagnostics = diagnose_run(part_dir)
    if kind == "xsa":
        assert all(layer["post"]["mean_abs"] < 2e-5 for layer in diagnostics["layers"])
    meta = json.loads((part_dir / "run_manifest.json").read_text())
    assert "mps" in meta["hardware"].lower()
    summary = json.loads((part_dir / "summary.json").read_text())
    assert summary["peak_device_bytes"] > 0
    assert summary["peak_mps_driver_bytes"] > 0
    assert "sampled" in summary["device_memory_measurement"]
    output = part_dir / "figures"
    plot_runs([part_dir], output)
    assert (output / "peak_device_memory_vs_tokens.png").stat().st_size > 5000
    assert (output / "mps_driver_memory_vs_tokens.png").stat().st_size > 5000


def test_real_mps_rng_restore():
    seed_all(42)
    saved = rng_state()
    expected = torch.rand(8, device="mps").cpu()
    restore_rng(saved)
    assert torch.equal(torch.rand(8, device="mps").cpu(), expected)
