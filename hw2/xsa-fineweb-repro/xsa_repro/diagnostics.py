"""Per-layer self-value cosines without modifying SDPA during training."""
import argparse
import json
from pathlib import Path

import torch

from .evaluate import load_run
from .utils import autocast_context, canonical_hash, write_json


class CosineAccumulator:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.sum, self.sum_square, self.abs_sum = 0.0, 0.0, 0.0
        self.valid, self.excluded_first, self.excluded_norm = 0, 0, 0

    def update(self, a, v):
        self.excluded_first += a.size(0) * a.size(1)
        a, v = a[..., 1:, :].float(), v[..., 1:, :].float()
        an, vn = a.norm(dim=-1), v.norm(dim=-1)
        good = (an > self.eps) & (vn > self.eps) & torch.isfinite(an) & torch.isfinite(vn)
        self.excluded_norm += good.numel() - good.sum().item()
        cos = ((a * v).sum(-1)[good] / (an[good] * vn[good])).cpu().double()
        self.sum += cos.sum().item()
        self.sum_square += cos.square().sum().item()
        self.abs_sum += cos.abs().sum().item()
        self.valid += cos.numel()

    def result(self):
        mean = self.sum / self.valid if self.valid else None
        std = max(0.0, self.sum_square / self.valid - mean ** 2) ** 0.5 if self.valid else None
        return {"mean": mean, "std": std, "mean_abs": self.abs_sum / self.valid if self.valid else None,
                "valid": self.valid, "excluded_first": self.excluded_first, "excluded_norm": self.excluded_norm,
                "norm_threshold": self.eps}


@torch.no_grad()
def diagnose_model(model, stream, batch_size, device, precision="fp32", max_windows=32):
    if batch_size <= 0 or (max_windows is not None and max_windows <= 0):
        raise ValueError("Diagnostic batch/window counts must be positive")
    stats = [{"pre": CosineAccumulator(), "post": CosineAccumulator()} for _ in model.blocks]
    previous_hooks = [block.attn.diagnostic_hook for block in model.blocks]
    was_training = model.training
    model.eval()
    for block, layer in zip(model.blocks, stats):
        def hook(y, z, v, layer=layer):
            layer["pre"].update(y, v)
            if model.attention_type == "xsa":
                layer["post"].update(z, v)
        block.attn.diagnostic_hook = hook
    starts = stream.validation_starts(max_windows)
    try:
        for offset in range(0, len(starts), batch_size):
            x, _ = stream.batch(starts[offset:offset + batch_size], device)
            with autocast_context(device, precision):
                model(x)
    finally:
        for block, hook in zip(model.blocks, previous_hooks):
            block.attn.diagnostic_hook = hook
        model.train(was_training)
    layers = [{"layer": i, "pre": s["pre"].result(),
               "post": s["post"].result() if model.attention_type == "xsa" else None} for i, s in enumerate(stats)]
    return {"attention": model.attention_type, "layers": layers, "windows": len(starts),
            "window_starts_sha256": canonical_hash(starts.tolist()), "precision": precision,
            "projection_fp32": model.config.xsa_projection_fp32,
            "interpretation": "Post-XSA cosine near zero checks the projection algebra; it is not evidence of lower perplexity."}


def diagnose_run(run_dir, checkpoint="final.pt", device_override=None, max_windows=None):
    model, stream, meta, state, device, precision = load_run(run_dir, checkpoint, device_override)
    t = meta["config"]["train"]
    result = diagnose_model(model, stream, t["eval_batch_size"], device, precision,
                            t["diagnostic_max_windows"] if max_windows is None else max_windows)
    result.update(tokens_seen=state["tokens_seen"], checkpoint=checkpoint, dataset_sha=meta["dataset_sha"])
    name = "diagnostics.json" if checkpoint == "final.pt" else f"diagnostics_{Path(checkpoint).stem}.json"
    write_json(Path(run_dir) / name, result)
    print(json.dumps(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument("--device")
    p.add_argument("--max-windows", type=int)
    args = p.parse_args()
    diagnose_run(args.run_dir, args.checkpoint, args.device, args.max_windows)


if __name__ == "__main__":
    main()
