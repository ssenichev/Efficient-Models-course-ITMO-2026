"""Read-only token streams and reproducible sampling independent of model RNG."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .utils import canonical_hash, file_hash


class TokenStream:
    def __init__(self, path, block_size, vocab_size):
        path = Path(path)
        if not path.exists() or path.stat().st_size % 2:
            raise ValueError(f"Missing or malformed uint16 stream: {path}")
        if path.stat().st_size < 2 * (block_size + 1):
            raise ValueError("Stream needs at least block_size + 1 tokens")
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.block_size = block_size
        if int(self.tokens.max()) >= vocab_size:
            raise ValueError("Token ID outside model vocabulary")
        self.num_windows = (len(self.tokens) - 1) // block_size

    def batch(self, starts, device="cpu"):
        starts = np.asarray(starts, dtype=np.int64)
        if starts.ndim != 1 or not len(starts):
            raise ValueError("Expected nonempty 1-D window starts")
        if np.any(starts < 0) or np.any(starts + self.block_size >= len(self.tokens)):
            raise IndexError("Window exceeds token stream")
        x = np.stack([self.tokens[s:s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.tokens[s + 1:s + self.block_size + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def validation_starts(self, max_windows=None):
        count = self.num_windows if max_windows is None else min(self.num_windows, max_windows)
        return np.arange(count, dtype=np.int64) * self.block_size


class WindowSampler:
    """Uniform, with replacement, over packed windows at multiples of T."""
    def __init__(self, stream, seed):
        self.num_windows, self.block_size = stream.num_windows, stream.block_size
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.windows_drawn = 0

    def sample(self, batch_size):
        starts = self.rng.integers(0, self.num_windows, size=batch_size) * self.block_size
        self.windows_drawn += batch_size
        return starts

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "num_windows": self.num_windows,
                "block_size": self.block_size, "windows_drawn": self.windows_drawn}

    def load_state_dict(self, state):
        if (state["num_windows"], state["block_size"]) != (self.num_windows, self.block_size):
            raise ValueError("Sampler corpus/context changed")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.windows_drawn = state["windows_drawn"]


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for split in ("train", "val"):
        entry = manifest["splits"][split]
        path = directory / f"{split}.bin"
        if path.stat().st_size != entry["tokens"] * 2 or file_hash(path) != entry["sha256"]:
            raise ValueError(f"{split} binary differs from manifest")
        docs = directory / f"{split}.documents.jsonl"
        if file_hash(docs) != entry["document_index_sha256"]:
            raise ValueError(f"{split} document index differs from manifest")
    hashes = []
    for split in ("train", "val"):
        with (directory / f"{split}.documents.jsonl").open() as f:
            hashes.append([json.loads(line)["sha256"] for line in f])
        if len(hashes[-1]) != manifest["splits"][split]["documents"]:
            raise ValueError("Document count differs from manifest")
    if any(len(h) != len(set(h)) for h in hashes) or set(hashes[0]) & set(hashes[1]):
        raise ValueError("Duplicate document hash or split leakage")
    if file_hash(directory / "document_audit.sqlite") != manifest["document_audit_sha256"]:
        raise ValueError("Document audit differs from manifest")
    return manifest, canonical_hash(manifest)
