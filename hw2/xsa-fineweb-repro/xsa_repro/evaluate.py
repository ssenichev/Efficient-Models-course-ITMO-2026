"""Fixed non-overlapping validation windows and token-weighted NLL."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import TokenStream, verify_manifest
from .model import GPT, ModelConfig
from .utils import autocast_context, file_hash, resolve_device_precision, write_json


@torch.no_grad()
def evaluate_model(model, stream, batch_size, device, precision="fp32", max_windows=None):
    if batch_size <= 0 or (max_windows is not None and max_windows <= 0):
        raise ValueError("Evaluation batch/window counts must be positive")
    was_training = model.training
    model.eval()
    starts = stream.validation_starts(max_windows)
    if not len(starts):
        raise ValueError("No validation windows")
    total_loss, total_tokens = 0.0, 0
    try:
        for offset in range(0, len(starts), batch_size):
            x, y = stream.batch(starts[offset:offset + batch_size], device)
            with autocast_context(device, precision):
                logits, _ = model(x)
            # Accumulate CE sums, never an unweighted mean of batch means.
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            # item() is a Python double; a device-side float64 cast is unsupported on MPS.
            total_loss += loss.item()
            total_tokens += y.numel()
    finally:
        model.train(was_training)
    nll = total_loss / total_tokens
    if not math.isfinite(nll):
        raise FloatingPointError("Non-finite validation NLL")
    return {"val_loss": nll, "val_ppl": math.exp(nll) if nll < 700 else None,
            "evaluated_tokens": total_tokens, "evaluated_windows": len(starts),
            "ignored_tail_tokens": len(stream.tokens) - 1 - stream.num_windows * stream.block_size}


def load_run(run_dir, checkpoint="final.pt", device_override=None):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_manifest.json").read_text())
    config = meta["config"]
    train = dict(config["train"])
    if device_override:
        train.update(device=device_override, precision="auto")
    device, precision = resolve_device_precision(train)
    torch.set_num_threads(train["cpu_threads"])
    # Trust only local checkpoints produced by this project: RNG state needs full pickle.
    state = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    if state["config_sha"] != meta["config_sha"] or state["dataset_sha"] != meta["dataset_sha"]:
        raise ValueError("Checkpoint provenance differs from run manifest")
    _, dataset_sha = verify_manifest(meta["data_dir"])
    if dataset_sha != meta["dataset_sha"]:
        raise ValueError("Corpus manifest changed since training")
    model = GPT(ModelConfig(**config["model"]), meta["attention"]).to(device)
    model.load_state_dict(state["model"])
    stream = TokenStream(Path(meta["data_dir"]) / "val.bin", config["model"]["block_size"], config["model"]["vocab_size"])
    return model, stream, meta, state, device, precision


def evaluate_run(run_dir, full_val=False, checkpoint="final.pt", device_override=None):
    model, stream, meta, state, device, precision = load_run(run_dir, checkpoint, device_override)
    config = meta["config"]
    result = evaluate_model(model, stream, config["train"]["eval_batch_size"], device, precision,
                            None if full_val else config["train"]["eval_max_windows"])
    result.update(checkpoint=checkpoint, checkpoint_sha256=file_hash(Path(run_dir) / checkpoint),
                  tokens_seen=state["tokens_seen"], step=state["step"], full_val=full_val,
                  attention=meta["attention"], precision=precision, dataset_sha=meta["dataset_sha"])
    name = "evaluation_final.json" if full_val and checkpoint == "final.pt" else f"evaluation_{Path(checkpoint).stem}_{'full' if full_val else 'subset'}.json"
    write_json(Path(run_dir) / name, result)
    print(json.dumps(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--full-val", action="store_true")
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    evaluate_run(args.run_dir, args.full_val, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
