import pytest
import torch
from xsa_repro.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


@pytest.fixture
def smoke_config(tmp_path):
    from pathlib import Path
    c = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    c["data"]["processed_dir"] = str(tmp_path / "data")
    c["runs_dir"] = str(tmp_path / "runs")
    return c
