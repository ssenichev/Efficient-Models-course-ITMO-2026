"""YAML inheritance and early validation of the experimental protocol."""
import copy
import math
from pathlib import Path
import yaml

from .model import ModelConfig


def merge(base, overrides):
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError("Cyclic config inheritance")
    seen.add(path)
    with path.open() as f:
        config = yaml.safe_load(f)
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, seen), config)
    validate(config)
    return config


def training_plan(config):
    t, length = config["train"], config["model"]["block_size"]
    micro_tokens = t["micro_batch_size"] * length
    if t["global_batch_tokens"] <= 0 or t["global_batch_tokens"] % micro_tokens:
        raise ValueError("global_batch_tokens must be divisible by micro_batch_size * block_size")
    return {"grad_accum_steps": t["global_batch_tokens"] // micro_tokens,
            "total_steps": math.ceil(t["train_tokens_budget"] / t["global_batch_tokens"]),
            "actual_tokens_budget": math.ceil(t["train_tokens_budget"] / t["global_batch_tokens"]) * t["global_batch_tokens"]}


def validate(config):
    ModelConfig(**config["model"])
    t, d = config["train"], config["data"]
    for field in ("micro_batch_size", "train_tokens_budget", "eval_every_steps", "eval_batch_size",
                  "checkpoint_every_steps", "log_every_steps", "diagnostic_max_windows", "cpu_threads"):
        if t[field] <= 0:
            raise ValueError(f"{field} must be positive")
    if t["warmup_steps"] < 0 or t["learning_rate"] <= 0 or not 0 <= t["min_lr_ratio"] <= 1:
        raise ValueError("Invalid learning rate schedule")
    if t["optimizer"] != "adamw" or t["lr_schedule"] != "warmup_cosine":
        raise ValueError("Only AdamW and warmup_cosine are supported")
    if not 0 < d["val_fraction_documents"] < 1 or min(d["train_tokens_target"], d["val_tokens_target"]) <= 0:
        raise ValueError("Invalid corpus targets or split fraction")
    if not 0 <= d["eos_id"] < config["model"]["vocab_size"] <= 65536:
        raise ValueError("Invalid uint16 vocabulary/EOT")
    if t["precision"] not in ("auto", "fp32", "bf16", "fp16"):
        raise ValueError("Invalid precision")
    if t["eval_max_windows"] is not None and t["eval_max_windows"] <= 0:
        raise ValueError("eval_max_windows must be positive or null")
    if d["shuffle_buffer_size"] <= 0 or t["grad_clip_norm"] <= 0:
        raise ValueError("Shuffle buffer and gradient clipping must be positive")
    if len(t["betas"]) != 2 or any(not 0 <= b < 1 for b in t["betas"]) or t["weight_decay"] < 0:
        raise ValueError("Invalid AdamW parameters")
    training_plan(config)
