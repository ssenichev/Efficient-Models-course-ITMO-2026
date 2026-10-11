"""Generate actual measured curves, paired summaries, and a cautious report."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xsa-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import write_json


def collect_run(path):
    path = Path(path)
    meta = json.loads((path / "run_manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    evaluation = json.loads((path / "evaluation_final.json").read_text())
    diagnostics = json.loads((path / "diagnostics.json").read_text())
    if not evaluation["full_val"] or evaluation["checkpoint"] != "final.pt":
        raise ValueError("Main summary requires full validation on final checkpoint")
    if evaluation["tokens_seen"] != summary["tokens_final"] or diagnostics["tokens_seen"] != summary["tokens_final"]:
        raise ValueError("Evaluation/diagnostics token count differs from final summary")
    if evaluation["dataset_sha"] != meta["dataset_sha"] or diagnostics["dataset_sha"] != meta["dataset_sha"]:
        raise ValueError("Analysis corpus differs from training corpus")
    if diagnostics["checkpoint"] != "final.pt" or diagnostics["attention"] != summary["attention"]:
        raise ValueError("Main diagnostics require the matching final checkpoint")
    summary.update(val_loss=evaluation["val_loss"], ppl=evaluation["val_ppl"], evaluated_tokens=evaluation["evaluated_tokens"],
                   evaluation_precision=evaluation["precision"], run_dir=str(path))
    logs = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    return {"summary": summary, "meta": meta, "logs": logs, "diagnostics": diagnostics}


def check_pairs(runs):
    grouped = {}
    for run in runs:
        s = run["summary"]
        key = (run["meta"]["config"]["experiment"], s["seed"], s["block_size"])
        group = grouped.setdefault(key, {})
        if s["attention"] in group:
            raise ValueError("Duplicate run for the same method/seed/context")
        group[s["attention"]] = run
    comparisons = []
    for key, group in grouped.items():
        if set(group) != {"sa", "xsa"}:
            continue
        sa, xsa = group["sa"], group["xsa"]
        for field in ("tokens_final", "dataset_sha", "config_sha", "initial_state_sha256", "params", "precision", "evaluation_precision", "evaluated_tokens"):
            if sa["summary"][field] != xsa["summary"][field]:
                raise ValueError(f"Unfair SA/XSA pair: {field} differs for {key}")
        if sa["diagnostics"]["window_starts_sha256"] != xsa["diagnostics"]["window_starts_sha256"]:
            raise ValueError("Diagnostics use different windows")
        if sa["meta"]["code"]["source_sha256"] != xsa["meta"]["code"]["source_sha256"]:
            raise ValueError("Paired runs use different source versions")
        comparisons.append({"experiment": key[0], "seed": key[1], "block_size": key[2],
                            "tokens_final": sa["summary"]["tokens_final"],
                            "delta_nll_sa_minus_xsa": sa["summary"]["val_loss"] - xsa["summary"]["val_loss"],
                            "synthetic": sa["summary"]["synthetic"]})
    return comparisons


def plot_runs(run_dirs, out_dir):
    runs = [collect_run(path) for path in run_dirs]
    comparisons = check_pairs(runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    synthetic = any(run["summary"]["synthetic"] for run in runs)
    prefix = "SYNTHETIC SOFTWARE SMOKE — " if synthetic else ""
    charts = [
        ("val_loss", "val_loss_vs_tokens.png", "Validation NLL (nats/token)"),
        ("train_loss", "train_loss_vs_tokens.png", "Train NLL (last optimizer batch, no smoothing)"),
        ("val_ppl", "perplexity_vs_tokens.png", "Validation perplexity"),
        ("throughput_tokens_per_sec", "throughput_vs_tokens.png", "Train tokens/s (step, synchronized accelerator)"),
        ("peak_vram_bytes", "peak_vram_vs_tokens.png", "CUDA peak allocated bytes (0 on non-CUDA)"),
    ]
    if any(run["summary"].get("device") == "mps" for run in runs):
        charts.append(("peak_device_bytes", "peak_device_memory_vs_tokens.png", "MPS sampled allocated bytes (lower bound on peak)"))
        charts.append(("peak_mps_driver_bytes", "mps_driver_memory_vs_tokens.png", "MPS sampled driver bytes (lower bound on peak)"))
    for field, filename, ylabel in charts:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for run in runs:
            s = run["summary"]
            # Full-val row replaces subset evaluation at final tokens_seen.
            points = {r["tokens_seen"]: r[field] for r in run["logs"] if r.get(field) is not None}
            if field in ("val_loss", "val_ppl"):
                points[s["tokens_final"]] = s["val_loss" if field == "val_loss" else "ppl"]
            points = sorted(points.items())
            if points:
                ax.plot(*zip(*points), marker="o", markersize=3,
                        label=f'{s["attention"].upper()} seed={s["seed"]} T={s["block_size"]}')
        ax.set(xlabel="Processed training tokens", ylabel=ylabel, title=textwrap.fill(prefix + ylabel, width=68))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for run in runs:
        s, layers = run["summary"], run["diagnostics"]["layers"]
        for phase in ("pre", "post"):
            if s["attention"] == "sa" and phase == "post":
                continue
            values = [layer[phase]["mean"] for layer in layers]
            ax.plot([layer["layer"] for layer in layers], [np.nan if v is None else v for v in values],
                    marker="o", linestyle="--" if phase == "post" else "-",
                    label=f'{s["attention"].upper()} {phase} seed={s["seed"]} T={s["block_size"]}')
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="Layer (zero-based)", ylabel="Mean cosine with self-value",
           title=textwrap.fill(prefix + "Self-value cosine; first position and small norms excluded", width=68))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "cosine_similarity_by_layer.png", dpi=150)
    plt.close(fig)
    summaries = [run["summary"] for run in runs]
    fields = list(dict.fromkeys(k for s in summaries for k in s))
    for filename in ("training_summary.csv", "summary.csv"):
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
    aggregates = []
    for experiment, context, tokens in sorted({(p["experiment"], p["block_size"], p["tokens_final"]) for p in comparisons}):
        values = [p["delta_nll_sa_minus_xsa"] for p in comparisons
                  if (p["experiment"], p["block_size"], p["tokens_final"]) == (experiment, context, tokens)]
        aggregates.append({"experiment": experiment, "block_size": context, "tokens_final": tokens,
                           "paired_seeds": len(values), "delta_nll_mean": float(np.mean(values)),
                           "delta_nll_std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    if len({a["block_size"] for a in aggregates}) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for a in aggregates:
            ax.errorbar(a["block_size"], a["delta_nll_mean"], yerr=a["delta_nll_std"] or 0.0,
                        fmt="o", capsize=4, label=f'{a["experiment"]}: {a["tokens_final"]} tokens')
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set(xlabel="Training context length", ylabel="Final ΔNLL (SA − XSA)", title=prefix + "Paired context ablation")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "paired_delta_nll_by_context.png", dpi=150)
        plt.close(fig)
    results = {"runs": summaries, "paired_comparisons": comparisons, "paired_seed_aggregates": aggregates,
               "synthetic": synthetic, "smoothing": "none", "primary_checkpoint": "final", "unpaired_runs": len(runs) - 2 * len(comparisons)}
    write_json(out / "results.json", results)
    write_json(out / "summary.json", results)
    lines = ["# XSA small-scale reproduction: measured run report", "",
             "**Synthetic software smoke only; these numbers do not test the scientific hypothesis.**" if synthetic
             else "Small-scale reproduction; not a replication of the original 100B-token experiments.", "",
             "## Method and protocol", "",
             "Original GPT implementation: RoPE on Q/K, embedding LayerNorm, pre-norm GELU decoder, tied embeddings. "
             "XSA removes the per-head projection on the current value after causal SDPA and before output projection. "
             "Paired runs are checked for equal initial weights, configuration, dataset, final token budget and evaluated tokens.", "",
             "The primary comparison uses final checkpoints at equal processed tokens. Validation is token-weighted NLL; "
             "PPL = exp(NLL). Intermediate curves use the fixed subset; the last point uses full validation. "
             "Training curves show the last optimizer batch at each logged step without smoothing.", "",
             "## Actual measurements", "", "| Method | Seed | T | Params | Tokens | Final NLL | PPL | Train tok/s | Accelerator bytes* |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f'| {s["attention"]} | {s["seed"]} | {s["block_size"]} | {s["params"]} | {s["tokens_final"]} | {s["val_loss"]:.6f} | {s["ppl"]:.4f} | {s["tokens_s"]:.1f} | {s.get("peak_device_bytes", s["peak_gpu_bytes"])} |')
    lines += ["", "*CUDA: true peak tensor allocation. MPS: maximum sampled current allocation after updates/evaluation, a lower bound on peak. CPU: not applicable (0).",
              "", "## Paired results", ""]
    for p in comparisons:
        lines.append(f'- T={p["block_size"]}, seed={p["seed"]}: ΔNLL (SA − XSA) = {p["delta_nll_sa_minus_xsa"]:.6f}.')
    if not comparisons:
        lines.append("No complete comparable SA/XSA pair was supplied; no paired conclusion is available.")
    for a in aggregates:
        if a["paired_seeds"] > 1:
            lines.append(f'- T={a["block_size"]}: paired ΔNLL mean ± sample std = {a["delta_nll_mean"]:.6f} ± {a["delta_nll_std"]:.6f} across {a["paired_seeds"]} seeds.')
    lines += ["", "## Cosine diagnostics", "",
              "Near-zero post-XSA cosine is a mathematical sanity check, not evidence of improved perplexity. "
              "Signed mean, mean absolute cosine, dispersion and excluded counts are saved in cosine_statistics.json.", "",
              "## Scope and limitations", "",
              "The planned main experiment is 44.65M total parameters, sample-10BT source, context 1024, ~100M processed tokens. "
              "The paper uses much larger models, sample-100BT, context 2048, and ~100B tokens. "
              "FineWeb's bounded streaming shuffle is not a global uniform shuffle. Normalized-content hashes split documents "
              "before packing; windows may cross EOT boundaries. Sampling is with replacement over packed training windows. "
              "Validation ignores the tail shorter than a full window; evaluated counts are saved.", "",
              "GPU speed/memory and BF16/FP16 behavior require measurements on the target GPU. Hardware, versions, precision, "
              "projection precision, source hashes, dataset hashes and initialization hashes are recorded per run. "
              "Small samples, single seeds, learning rate selection and different corpus composition limit inference. "
              "Negative ΔNLL is a valid experimental result; do not infer the paper's large-scale result from a smoke run.", "",
              "## Sources", "",
              "- [Zhai, Exclusive Self Attention](https://arxiv.org/html/2603.09078v1)",
              "- [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)",
              "- [nanoGPT](https://github.com/karpathy/nanoGPT)", ""]
    (out / "report_draft.md").write_text("\n".join(lines))
    write_json(out / "cosine_statistics.json", {r["summary"]["run_dir"]: r["diagnostics"] for r in runs})
    print(json.dumps({"out_dir": str(out), "runs": len(runs), "pairs": len(comparisons), "synthetic": synthetic}))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", required=True, nargs="+")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    plot_runs(args.run_dirs, args.out_dir)


if __name__ == "__main__":
    main()
