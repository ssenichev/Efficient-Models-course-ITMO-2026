"""Measure latency, peak memory and energy of SmallCNN over the (S, B) grid.

    python measure.py                 # full grid -> results/measurements.csv
    python measure.py --quick         # few configs, smoke test
"""
import argparse
import json
import math
import os
import threading
import time

import numpy as np
import pandas as pd
import torch

from models import build_model

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def make_grid(seed):
    rng = np.random.default_rng(seed)
    s_pool = [s for s in range(32, 513, 16) if s not in BASE_S]
    b_pool = [b for b in range(1, 257) if b & (b - 1)]  # not a power of two
    extra_s = sorted(int(s) for s in rng.choice(s_pool, 4, replace=False))
    extra_b = sorted(int(b) for b in rng.choice(b_pool, 3, replace=False))
    return extra_s, extra_b


class EnergyMeter:
    """Whole-GPU energy via NVML. Uses the hardware energy counter when the GPU
    has one (Volta and newer, e.g. T4), otherwise integrates power samples."""

    def __init__(self, index=0, period=0.005):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.period = period
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.mode = "counter"
        except pynvml.NVMLError:
            self.mode = "sampling"

    def _power(self):
        return self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0  # W

    def _sample(self):
        t_prev, p_prev = time.perf_counter(), self._power()
        while not self._stop.is_set():
            time.sleep(self.period)
            t, p = time.perf_counter(), self._power()
            self._joules += 0.5 * (p + p_prev) * (t - t_prev)
            t_prev, p_prev = t, p

    def start(self):
        if self.mode == "counter":
            self._e0 = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h)
        else:
            self._joules = 0.0
            self._stop = threading.Event()
            self._thr = threading.Thread(target=self._sample, daemon=True)
            self._thr.start()

    def stop(self):
        if self.mode == "counter":
            return (self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) - self._e0) / 1000.0
        self._stop.set()
        self._thr.join()
        return self._joules

    def status(self):
        clk = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
        temp = self.nv.nvmlDeviceGetTemperature(self.h, self.nv.NVML_TEMPERATURE_GPU)
        return clk, temp


def peak_memory(model, x):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        y = model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del y
    return peak


def time_forward(model, x, budget=0.5, min_reps=15, max_reps=200, warmup=3):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        est = time.perf_counter() - t0
        reps = int(np.clip(budget / est, min_reps, max_reps))
        times = np.empty(reps)
        for i in range(reps):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times[i] = time.perf_counter() - t0
    return float(np.median(times)), times


def energy_forward(model, x, meter, lat, window):
    n = max(3, math.ceil(window / lat))
    with torch.inference_mode():
        torch.cuda.synchronize()
        meter.start()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        joules = meter.stop()
    return joules / n, joules / dt, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/measurements.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--energy-window", type=float, default=1.5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)

    extra_s, extra_b = make_grid(args.seed)
    all_s, all_b = sorted(BASE_S + extra_s), sorted(BASE_B + extra_b)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(os.path.join(os.path.dirname(args.out), "grid.json"), "w") as f:
        json.dump({"seed": args.seed, "extra_S": extra_s, "extra_B": extra_b}, f, indent=2)
    print("extra S:", extra_s, " extra B:", extra_b)

    configs = [(s, b) for s in all_s for b in all_b]
    if args.quick:
        configs = [(32, 1), (128, 8), (224, 32), (extra_s[0], extra_b[0])]
    # random order so that slow drifts (temperature, clocks) don't correlate with S or B
    order = np.random.default_rng(args.seed + 1).permutation(len(configs))
    configs = [configs[i] for i in order]

    done = set()
    if os.path.exists(args.out) and not args.no_resume and not args.quick:
        old = pd.read_csv(args.out)
        done = set(zip(old.S, old.B))
    elif os.path.exists(args.out):
        os.remove(args.out)

    props = torch.cuda.get_device_properties(0)
    env = dict(gpu=props.name, gpu_memory_bytes=props.total_memory,
               torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version())
    try:
        import pynvml
        pynvml.nvmlInit()
        env["driver"] = pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        pass
    with open(os.path.join(os.path.dirname(args.out), "env.json"), "w") as f:
        json.dump(env, f, indent=2, default=str)
    print(env)

    model = build_model().cuda().eval()
    try:
        meter = EnergyMeter(torch.cuda.current_device())
        print("energy meter:", meter.mode)
    except Exception as e:  # no NVML -> no energy
        print("NVML not available:", e)
        meter = None

    for k, (S, B) in enumerate(configs):
        if (S, B) in done:
            continue
        row = dict(S=S, B=B, is_validation=int(S in extra_s or B in extra_b),
                   latency_s=np.nan, latency_p10=np.nan, latency_p90=np.nan,
                   memory_bytes=np.nan, energy_j=np.nan, power_w=np.nan,
                   n_energy=0, sm_clock_mhz=np.nan, temp_c=np.nan, status="ok")
        x = None
        try:
            torch.cuda.empty_cache()
            x = torch.randn(B, 3, S, S, device="cuda")
            row["memory_bytes"] = peak_memory(model, x)
            lat, times = time_forward(model, x)
            row.update(latency_s=lat, latency_p10=np.percentile(times, 10),
                       latency_p90=np.percentile(times, 90))
            if meter is not None:
                e, p, n = energy_forward(model, x, meter, lat, args.energy_window)
                row.update(energy_j=e, power_w=p, n_energy=n)
                row["sm_clock_mhz"], row["temp_c"] = meter.status()
        except torch.cuda.OutOfMemoryError:
            row.update(status="OOM", memory_bytes="OOM")
        except RuntimeError as e:
            if "out of memory" in str(e):
                row.update(status="OOM", memory_bytes="OOM")
            else:
                row["status"] = "error: " + str(e).splitlines()[0][:80]
        finally:
            del x
            torch.cuda.empty_cache()

        pd.DataFrame([row]).to_csv(args.out, mode="a", index=False,
                                   header=not os.path.exists(args.out))
        mem = row["memory_bytes"]
        mem = mem if isinstance(mem, str) else f"{mem / 2**20:8.1f} MiB"
        print(f"[{k + 1:3d}/{len(configs)}] S={S:3d} B={B:3d}  "
              f"lat={row['latency_s'] * 1e3:9.3f} ms  mem={mem}  "
              f"E={row['energy_j']:.4g} J  {row['status']}", flush=True)


if __name__ == "__main__":
    main()
