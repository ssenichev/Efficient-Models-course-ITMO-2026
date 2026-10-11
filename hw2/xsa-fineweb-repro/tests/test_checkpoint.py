import copy
import random
import numpy as np
import pytest
import torch

from xsa_repro.prepare_data import prepare
from xsa_repro.train import train_run, WarmupCosine
from xsa_repro.utils import restore_rng, rng_state, seed_all


@pytest.mark.parametrize("kind", ["sa", "xsa"])
def test_resume_matches_uninterrupted(smoke_config, kind):
    prepare(smoke_config)
    full_config = copy.deepcopy(smoke_config)
    full_config["experiment"] = "full"
    full_dir = train_run(full_config, kind)
    part_config = copy.deepcopy(smoke_config)
    part_config["experiment"] = "resume"
    part_dir = train_run(part_config, kind, max_steps=2)
    assert not (part_dir / "final.pt").exists()
    initial = torch.load(part_dir / "initial.pt", weights_only=False)
    full_initial = torch.load(full_dir / "initial.pt", weights_only=False)
    assert all(torch.equal(v, full_initial["model"][k]) for k, v in initial["model"].items())
    checkpoint = torch.load(part_dir / "latest.pt", weights_only=False)
    assert checkpoint["step"] == 2 and checkpoint["tokens_seen"] == 128
    assert {"rng", "sampler", "optimizer", "scheduler", "scaler"} <= set(checkpoint)
    train_run(part_config, kind, resume=part_dir / "latest.pt")
    full = torch.load(full_dir / "final.pt", weights_only=False)
    resumed = torch.load(part_dir / "final.pt", weights_only=False)
    assert full["tokens_seen"] == resumed["tokens_seen"] == 256
    assert full["step"] == resumed["step"] == 4
    assert full["sampler"] == resumed["sampler"]
    assert full["scheduler"] == resumed["scheduler"]
    assert torch.equal(full["rng"]["torch"], resumed["rng"]["torch"])
    for key, value in full["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    for key, state in full["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)
    wrong = copy.deepcopy(part_config)
    wrong["train"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        train_run(wrong, kind, resume=part_dir / "latest.pt")


def test_rng_roundtrip():
    seed_all(42)
    saved = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(saved)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_lr_endpoints():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1)
    scheduler = WarmupCosine(optimizer, 10, 2, 0.001, 0.1)
    assert scheduler.set_step(1) == 0.0005
    assert scheduler.set_step(2) == 0.001
    assert scheduler.set_step(10) == pytest.approx(0.0001)


def test_rewind_to_initial(smoke_config):
    prepare(smoke_config)
    run = train_run(smoke_config, "xsa")
    original = torch.load(run / "final.pt", weights_only=False)
    train_run(smoke_config, "xsa", resume=run / "initial.pt", max_steps=2)
    assert not (run / "final.pt").exists()
    assert (run / "final.pt.stale").exists()
    best = torch.load(run / "best_val.pt", weights_only=False)
    assert best["step"] <= 2
    train_run(smoke_config, "xsa", resume=run / "latest.pt")
    rewound = torch.load(run / "final.pt", weights_only=False)
    assert all(torch.equal(v, rewound["model"][k]) for k, v in original["model"].items())


def test_changed_split_is_rejected(smoke_config):
    prepare(smoke_config)
    smoke_config["data"]["split_salt"] = "different-salt"
    with pytest.raises(ValueError, match="data configuration"):
        train_run(smoke_config, "sa")
