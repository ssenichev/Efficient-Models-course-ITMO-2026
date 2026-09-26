"""Fit latency and energy parameters on the base grid, report errors on the
held-out (randomly sampled) configurations.

    python calibrate.py   # reads results/measurements.csv, writes results/theta.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

import equations as eq


def load(path):
    df = pd.read_csv(path)
    df["oom"] = df["status"].eq("OOM")
    df["memory_bytes"] = pd.to_numeric(df["memory_bytes"], errors="coerce")
    return df.sort_values(["S", "B"]).reset_index(drop=True)


def fit_latency(S, B, t):
    # parameters in log space: they span ~15 orders of magnitude together
    def theta_of(p):
        return dict(t_op=np.exp(p[0]), peak_flops=np.exp(p[1]), bandwidth=np.exp(p[2]))

    def resid(p):
        return np.log(eq.latency(S, B, theta_of(p))) - np.log(t)

    p0 = np.log([1e-5, 4e12, 250e9])
    best = None
    # max() makes the loss piecewise, so try a few starts
    for scale in [(1, 1, 1), (0.3, 0.5, 0.5), (3, 2, 2), (1, 0.25, 1)]:
        r = least_squares(resid, p0 + np.log(scale), loss="soft_l1", f_scale=0.1)
        if best is None or r.cost < best.cost:
            best = r
    return theta_of(best.x)


def fit_energy(S, B, e, t):
    # E = p_static * t + e_flop * F + e_byte * Q, linear and non-negative.
    # Rows are divided by E so that small and large configs weigh the same.
    X = np.stack([t, eq.flops(S, B), eq.bytes_moved(S, B)], axis=1)
    col = X.max(axis=0)
    coef, _ = nnls(X / col / e[:, None], np.ones_like(e))
    coef = coef / col
    return dict(p_static=coef[0], e_flop=coef[1], e_byte=coef[2])


def ape(pred, meas):
    return np.abs(pred - meas) / meas * 100


def summary(name, pred, meas, mask_tr, mask_va):
    err = ape(pred, meas)
    out = {}
    for tag, m in [("train", mask_tr), ("val", mask_va)]:
        if m.sum():
            out[tag] = dict(mape=float(np.mean(err[m])), median=float(np.median(err[m])),
                            max=float(np.max(err[m])), n=int(m.sum()))
    print(f"{name:8s}" + "".join(
        f"  {k}: MAPE {v['mape']:6.2f}%  median {v['median']:6.2f}%  max {v['max']:7.2f}%  (n={v['n']})"
        for k, v in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/measurements.csv")
    ap.add_argument("--out", default="results/theta.json")
    args = ap.parse_args()

    df = load(args.csv)
    ok = df["status"].eq("ok").to_numpy()
    tr = ok & (df["is_validation"] == 0).to_numpy()
    va = ok & (df["is_validation"] == 1).to_numpy()
    S, B = df["S"].to_numpy(float), df["B"].to_numpy(float)
    t = df["latency_s"].to_numpy()

    theta = fit_latency(S[tr], B[tr], t[tr])
    print("latency theta:", {k: f"{v:.4g}" for k, v in theta.items()})

    metrics = {}
    metrics["latency"] = summary("latency", eq.latency(S, B, theta), t, tr, va)
    metrics["memory"] = summary("memory", eq.memory(S, B), df["memory_bytes"].to_numpy(), tr, va)

    theta_e = None
    e = df["energy_j"].to_numpy()
    has_e = ~np.isnan(e)
    if (tr & has_e).sum() > 3:
        m = tr & has_e
        theta_e = fit_energy(S[m], B[m], e[m], t[m])
        theta_e["latency"] = theta
        print("energy theta:", {k: f"{v:.4g}" for k, v in theta_e.items() if k != "latency"})
        metrics["energy"] = summary("energy", eq.energy(S, B, theta_e), e, tr & has_e, va & has_e)

    # OOM: which configs the memory model says cannot fit
    n_oom = int(df["oom"].sum())
    print(f"OOM measured: {n_oom} configs")

    with open(args.out, "w") as f:
        json.dump({"latency": theta, "energy": theta_e, "metrics": metrics}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
