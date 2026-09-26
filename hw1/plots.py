"""Figures: measured points vs model predictions.

    python plots.py   # reads results/*.csv|json, writes results/figures/*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import equations as eq
from calibrate import load
from models import build_model

RES = "results"
FIG = os.path.join(RES, "figures")
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

S_LINE = np.linspace(32, 512, 200)
B_LINE = np.geomspace(1, 256, 200)


def count_flops(S, B):
    with torch.device("meta"):
        m = build_model().eval()
        x = torch.randn(B, 3, S, S)
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops()


def colors(values):
    cmap = plt.get_cmap("viridis")
    v = np.log2(np.asarray(values, float))
    v = (v - v.min()) / max(np.ptp(v), 1e-9)
    return [cmap(0.05 + 0.85 * t) for t in v]


def scatter_meas(ax, x, y, val, c, label=None):
    ax.scatter(x[~val], y[~val], color=c, s=22, zorder=3, label=label)
    ax.scatter(x[val], y[val], facecolors="white", edgecolors=[c], s=26, lw=1.3, zorder=3)


def legend_note(ax):
    ax.plot([], [], "o", color="gray", label="measured (base grid)")
    ax.plot([], [], "o", mfc="white", mec="gray", label="measured (held-out)")
    ax.plot([], [], "-", color="gray", label="model")


def vs_S(df, col, pred_fn, ylabel, fname, title, b_show=None, scale=1.0):
    b_show = b_show or [1, 4, 16, 64, 256]
    b_show = [b for b in sorted(df.B.unique()) if b in b_show or b not in eq_base_B()]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for b, c in zip(b_show, colors(b_show)):
        d = df[(df.B == b) & df[col].notna()]
        ls = "--" if b not in eq_base_B() else "-"
        ax.plot(S_LINE, pred_fn(S_LINE, b) * scale, ls, color=c, lw=1.6)
        scatter_meas(ax, d.S.to_numpy(), d[col].to_numpy() * scale,
                     d.is_validation.to_numpy() == 1, c, label=f"B={b}")
    legend_note(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([32, 64, 128, 256, 512], ["32", "64", "128", "256", "512"])
    ax.set_xlabel("image size S [px]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def vs_B(df, col, pred_fn, ylabel, fname, title, scale=1.0):
    s_show = [s for s in sorted(df.S.unique()) if s in (32, 128, 224, 512) or s not in eq_base_S()]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s, c in zip(s_show, colors(s_show)):
        d = df[(df.S == s) & df[col].notna()]
        ls = "--" if s not in eq_base_S() else "-"
        ax.plot(B_LINE, pred_fn(s, B_LINE) * scale, ls, color=c, lw=1.6)
        scatter_meas(ax, d.B.to_numpy(), d[col].to_numpy() * scale,
                     d.is_validation.to_numpy() == 1, c, label=f"S={s}")
    legend_note(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def eq_base_S():
    from measure import BASE_S
    return BASE_S


def eq_base_B():
    from measure import BASE_B
    return BASE_B


def surface(df, col, pred_fn, zlabel, fname, title):
    Sg, Bg = np.meshgrid(np.linspace(32, 512, 40), np.geomspace(1, 256, 40))
    Z = np.log10(pred_fn(Sg, Bg))
    d = df[df[col].notna()]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(Sg, np.log2(Bg), Z, cmap="viridis", alpha=0.45, lw=0)
    val = d.is_validation.to_numpy() == 1
    ax.scatter(d.S[~val], np.log2(d.B[~val]), np.log10(d[col][~val]), c="k", s=10, label="base grid")
    ax.scatter(d.S[val], np.log2(d.B[val]), np.log10(d[col][val]), c="red", s=12, label="held-out")
    ax.set_xlabel("S [px]")
    ax.set_ylabel("log2 B")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def parity(df, theta, theta_e):
    items = [("latency_s", lambda s, b: eq.latency(s, b, theta), "latency [s]"),
             ("memory_bytes", eq.memory, "peak memory [B]")]
    if theta_e:
        items.append(("energy_j", lambda s, b: eq.energy(s, b, theta_e), "energy [J]"))
    fig, axes = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.4))
    for ax, (col, fn, lab) in zip(axes, items):
        d = df[df[col].notna()]
        p, m = fn(d.S.to_numpy(float), d.B.to_numpy(float)), d[col].to_numpy()
        val = d.is_validation.to_numpy() == 1
        lo, hi = min(p.min(), m.min()) / 1.5, max(p.max(), m.max()) * 1.5
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.fill_between([lo, hi], [lo / 1.2, hi / 1.2], [lo * 1.2, hi * 1.2], color="gray", alpha=0.12, label="±20%")
        ax.scatter(m[~val], p[~val], s=14, color="C0", label="base grid")
        ax.scatter(m[val], p[val], s=16, facecolors="white", edgecolors="C3", label="held-out")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("measured " + lab); ax.set_ylabel("predicted " + lab)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parity.png"))
    plt.close(fig)


def error_map(df, theta, theta_e):
    cols = [("latency_s", lambda s, b: eq.latency(s, b, theta), "latency")]
    cols.append(("memory_bytes", eq.memory, "memory"))
    if theta_e:
        cols.append(("energy_j", lambda s, b: eq.energy(s, b, theta_e), "energy"))
    Ss, Bs = sorted(df.S.unique()), sorted(df.B.unique())
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4.6))
    for ax, (col, fn, name) in zip(axes, cols):
        E = np.full((len(Bs), len(Ss)), np.nan)
        for _, r in df[df[col].notna()].iterrows():
            E[Bs.index(r.B), Ss.index(r.S)] = (fn(r.S, r.B) - r[col]) / r[col] * 100
        lim = np.nanmax(np.abs(E)) if np.isfinite(E).any() else 1
        lim = min(lim, 50)
        im = ax.imshow(E, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        ax.set_xticks(range(len(Ss)), Ss, rotation=60, fontsize=7)
        ax.set_yticks(range(len(Bs)), Bs, fontsize=7)
        for i, b in enumerate(Bs):
            for j, s in enumerate(Ss):
                if df[(df.S == s) & (df.B == b)].status.eq("OOM").any():
                    ax.text(j, i, "OOM", ha="center", va="center", fontsize=5)
        ax.set_xlabel("S [px]"); ax.set_ylabel("B")
        ax.set_title(f"{name}: (pred − meas)/meas [%]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "error_map.png"))
    plt.close(fig)


def regimes(df, theta):
    # time split vs amount of work, and the regime map on the (S, B) plane
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for s, c in zip([32, 128, 512], ["C0", "C1", "C2"]):
        l, m, cmp = eq.latency_terms(s, B_LINE, theta)
        tot = l + m + cmp
        a1.plot(B_LINE, l / tot, ":", color=c)
        a1.plot(B_LINE, m / tot, "--", color=c)
        a1.plot(B_LINE, cmp / tot, "-", color=c, label=f"S={s}")
    a1.plot([], [], "k:", label="launch share"); a1.plot([], [], "k--", label="memory share")
    a1.plot([], [], "k-", label="compute share")
    a1.set_xscale("log", base=2); a1.set_xlabel("batch size B"); a1.set_ylabel("share of predicted latency")
    a1.set_title("where the time goes (model)"); a1.legend(fontsize=8, ncol=2)

    Sg, Bg = np.meshgrid(np.linspace(32, 512, 300), np.geomspace(1, 256, 300))
    R = eq.regime(Sg, Bg, theta)
    cm = matplotlib.colors.ListedColormap(["#d9d9d9", "#9ecae1", "#fdae6b"])
    a2.pcolormesh(Sg, Bg, R, cmap=cm, vmin=-0.5, vmax=2.5, shading="auto")
    d = df[df.status.eq("ok")]
    thr = d.B * 1.0 / d.latency_s
    sc = a2.scatter(d.S, d.B, c=np.log10(thr), cmap="viridis", s=18, edgecolors="k", lw=0.3)
    fig.colorbar(sc, ax=a2, label="log10 measured throughput [img/s]")
    for k, name in enumerate(["launch-bound", "memory-bound", "compute-bound"]):
        a2.fill_between([], [], color=cm(k), label=name)
    a2.set_xscale("log", base=2); a2.set_yscale("log", base=2)
    a2.set_xticks([32, 64, 128, 256, 512], ["32", "64", "128", "256", "512"])
    a2.set_xlabel("S [px]"); a2.set_ylabel("B")
    a2.set_title("dominant term of the latency model"); a2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "regimes.png"))
    plt.close(fig)

    # regime of every kernel vs batch size, for a small and a large image
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for ax, s0 in zip(axes, [32, 128, 512]):
        fl, by = eq.per_op(s0, B_LINE)
        K = np.zeros((len(eq.OPS), len(B_LINE)))
        for i, (f, q) in enumerate(zip(fl, by)):
            tc, tm = f / theta["peak_flops"], q / theta["bandwidth"]
            launch = (theta["t_op"] >= tc) & (theta["t_op"] >= tm)
            K[i] = np.where(launch, 0, np.where(tc >= tm, 2, 1))
        ax.pcolormesh(B_LINE, np.arange(len(eq.OPS)), K, cmap=cm,
                      vmin=-0.5, vmax=2.5, shading="nearest")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size B")
        ax.set_title(f"S = {s0}")
    axes[0].set_yticks(range(len(eq.OPS)), [o[0] for o in eq.OPS], fontsize=8)
    axes[0].invert_yaxis()
    for k, name in enumerate(["launch-bound", "memory-bound", "compute-bound"]):
        axes[-1].fill_between([], [], color=cm(k), label=name)
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("regime of every kernel in the fitted latency model")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "kernel_regimes.png"))
    plt.close(fig)

    # roofline of the whole network
    fig, ax = plt.subplots(figsize=(7, 5))
    ai = eq.flops(d.S, d.B) / eq.bytes_moved(d.S, d.B)
    perf = eq.flops(d.S, d.B) / d.latency_s
    x = np.geomspace(1, 100, 100)
    ax.plot(x, np.minimum(theta["peak_flops"], theta["bandwidth"] * x), "k-", label="fitted roofline")
    sc = ax.scatter(ai, perf, c=np.log2(d.B), s=18, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="log2 B")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity FLOPs / bytes [FLOP/B]")
    ax.set_ylabel("achieved FLOP/s (measured)")
    ax.set_title("whole-network roofline"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "roofline.png"))
    plt.close(fig)


def memory_plane(df, env):
    fig, ax = plt.subplots(figsize=(7, 5))
    cap = env.get("gpu_memory_bytes", 16e9)
    Sg, Bg = np.meshgrid(np.linspace(32, 512, 200), np.geomspace(1, 256, 200))
    cs = ax.contour(Sg, Bg, eq.memory(Sg, Bg) / 2**30, levels=[0.1, 0.5, 1, 2, 4, 8], colors="gray", linewidths=0.8)
    ax.clabel(cs, fmt="%g GiB", fontsize=7)
    if eq.memory(512, 256) > cap:
        ax.contour(Sg, Bg, eq.memory(Sg, Bg), levels=[cap], colors="red")
    ok = df[df.status.eq("ok")]
    oom = df[df.status.eq("OOM")]
    ax.scatter(ok.S, ok.B, s=12, color="C0", label="fits")
    ax.scatter(oom.S, oom.B, s=30, marker="x", color="red", label="OOM (measured)")
    ax.set_yscale("log", base=2)
    ax.set_xlim(16, 528); ax.set_ylim(0.8, 320)
    ax.set_xlabel("S [px]"); ax.set_ylabel("B")
    b_max = (cap - eq.WEIGHT_BYTES - eq.BLAS_WORKSPACE) / (68 * 512**2)
    ax.set_title(f"predicted peak memory; GPU has {cap / 2**30:.1f} GiB\n"
                 f"model: OOM at S=512 only if B > {b_max:.0f}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "memory_plane.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load(os.path.join(RES, "measurements.csv"))
    with open(os.path.join(RES, "theta.json")) as f:
        th = json.load(f)
    theta, theta_e = th["latency"], th["energy"]
    env = {}
    if os.path.exists(os.path.join(RES, "env.json")):
        env = json.load(open(os.path.join(RES, "env.json")))

    df["flops_counted"] = [count_flops(int(s), int(b)) for s, b in zip(df.S, df.B)]
    vs_S(df, "flops_counted", eq.flops, "FLOPs per forward pass", "flops.png",
         "FLOPs: FlopCounterMode (points) vs formula (lines)")
    vs_S(df, "memory_bytes", eq.memory, "peak allocated memory [MiB]", "memory_vs_S.png",
         "peak memory: measured vs model", scale=1 / 2**20)
    vs_B(df, "memory_bytes", eq.memory, "peak allocated memory [MiB]", "memory_vs_B.png",
         "peak memory: measured vs model", scale=1 / 2**20)
    memory_plane(df, env)

    lat = lambda s, b: eq.latency(s, b, theta)
    vs_S(df, "latency_s", lat, "latency [ms]", "latency_vs_S.png", "latency: measured vs model", scale=1e3)
    vs_B(df, "latency_s", lat, "latency [ms]", "latency_vs_B.png", "latency: measured vs model", scale=1e3)
    surface(df, "latency_s", lat, "log10 latency [s]", "latency_surface.png", "latency surface")
    regimes(df, theta)

    if theta_e:
        en = lambda s, b: eq.energy(s, b, theta_e)
        vs_S(df, "energy_j", en, "energy per forward [mJ]", "energy_vs_S.png", "energy: measured vs model", scale=1e3)
        vs_B(df, "energy_j", en, "energy per forward [mJ]", "energy_vs_B.png", "energy: measured vs model", scale=1e3)
        surface(df, "energy_j", en, "log10 energy [J]", "energy_surface.png", "energy surface")

    parity(df, theta, theta_e)
    error_map(df, theta, theta_e)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
