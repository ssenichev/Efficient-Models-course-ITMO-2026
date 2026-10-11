import pytest
import torch

from xsa_repro.benchmark import benchmark
from xsa_repro.config import load_config, training_plan
from xsa_repro.utils import resolve_device_precision, synchronize_device, device_memory


def test_auto_prefers_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device, precision = resolve_device_precision({"device": "auto", "precision": "auto"})
    assert device.type == "mps" and precision == "fp32"


def test_explicit_mps_errors_are_clear(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS unavailable"):
        resolve_device_precision({"device": "mps", "precision": "fp32"})
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(ValueError, match="Non-CUDA"):
        resolve_device_precision({"device": "mps", "precision": "fp16"})


def test_mps_synchronization_and_sampled_memory(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 123)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 456)
    synchronize_device(torch.device("mps"))
    assert calls == ["sync"]
    memory = device_memory(torch.device("mps"))
    assert memory["allocated_bytes"] == 123 and memory["driver_bytes"] == 456
    assert "lower bound" in memory["measurement"]


def test_mac_configs_keep_full_model():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    main = load_config(root / "configs/main.yaml")
    for filename, budget in (("mac_pilot.yaml", 1000000), ("mac_10m.yaml", 10000000)):
        config = load_config(root / "configs" / filename)
        for key in ("vocab_size", "n_layer", "n_embd", "n_head", "mlp_ratio"):
            assert config["model"][key] == main["model"][key]
        assert config["model"]["block_size"] == 256
        assert config["train"]["train_tokens_budget"] == budget
        assert training_plan(config)["grad_accum_steps"] == 16


def test_cpu_benchmark_is_short_and_finite(smoke_config):
    result = benchmark(smoke_config, "xsa", warmup_steps=0, steps=2)
    assert result["params"] == 33728 and result["tokens_s"] > 0
    assert result["synthetic"] and result["measured_steps"] == 2
    assert result["estimated_hours_one_model"]["10000000"] > result["estimated_hours_one_model"]["1000000"]
