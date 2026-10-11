"""Hashes, provenance, device selection, and complete RNG state."""
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "numpy", "datasets", "tiktoken", "huggingface-hub", "matplotlib", "PyYAML"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def code_provenance():
    root = Path(__file__).resolve().parent.parent
    files = sorted([*root.glob("xsa_repro/*.py"), *root.glob("configs/*.yaml"), root / "pyproject.toml"])
    hashes = {str(p.relative_to(root)): file_hash(p) for p in files}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"git_commit": commit, "source_sha256": canonical_hash(hashes), "files": hashes}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed, deterministic=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def resolve_device_precision(train):
    device = train.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Supported devices: cpu, cuda, mps")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable: check Apple Silicon/PyTorch and run outside a GPU-restricted sandbox")
    precision = train["precision"]
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("Non-CUDA experiments use fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16")
    return device, precision


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_memory(device):
    """CUDA high-water mark; MPS samples are explicitly a lower bound on peak."""
    if device.type == "cuda":
        return {"allocated_bytes": torch.cuda.max_memory_allocated(device), "driver_bytes": None,
                "measurement": "CUDA peak tensor allocation"}
    if device.type == "mps":
        return {"allocated_bytes": torch.mps.current_allocated_memory(),
                "driver_bytes": torch.mps.driver_allocated_memory(),
                "measurement": "MPS current allocation sampled after updates; lower bound on peak"}
    return {"allocated_bytes": 0, "driver_bytes": None, "measurement": "CPU; accelerator memory not applicable"}


def hardware_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                                           stderr=subprocess.DEVNULL).strip() + f" ({device.type})"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return f"{platform.machine()} ({device.type})"


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
