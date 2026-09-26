"""Closed-form cost model of SmallCNN (models.py).

All functions take image size S and batch B (scalars or numpy arrays, broadcast
against each other) and return floats/arrays.

Conventions
  FLOPs   : 1 MAC = 2 FLOPs, only conv and linear layers are counted
            (ReLU / pooling / bias adds are ~0.2% of the total).
  Bytes   : compulsory DRAM traffic per kernel = read inputs + read weights +
            write outputs, FP32 = 4 B, max-pool indices are int64 = 8 B.
  Memory  : bytes counted by torch.cuda.max_memory_allocated(): weights, the
            cuBLAS/cuBLASLt workspaces PyTorch keeps allocated after the first
            linear layer, the input and the live activations at the worst
            moment. cuDNN workspace is not modelled (it depends on the
            algorithm cuDNN picks for each shape).
  Latency : every kernel costs max(launch overhead, FLOPs/P, bytes/BW),
            the forward pass is the sum over kernels.
  Energy  : E = P_static * latency + e_flop * FLOPs + e_byte * bytes.
"""
import numpy as np

F32, I64 = 4, 8

# (cin, cout, k, input downsampling, output downsampling) relative to S
CONVS = [
    (3, 32, 7, 1, 2),
    (32, 64, 5, 4, 4),
    (64, 128, 3, 4, 8),
    (128, 256, 1, 8, 8),
    (256, 256, 3, 8, 16),
    (256, 512, 1, 16, 16),
]
FC = [(512, 256), (256, 100)]

N_PARAMS = sum(ci * co * k * k for ci, co, k, _, _ in CONVS) + sum(i * o + o for i, o in FC)
WEIGHT_BYTES = F32 * N_PARAMS  # 1 040 324 params -> 4 161 296 B

# PyTorch defaults on CUDA < sm90: cuBLAS workspace ":4096:2:16:8"
# (2 x 4 MiB + 8 x 16 KiB) plus a 1 MiB cuBLASLt workspace for addmm
BLAS_WORKSPACE = 2 * 4096 * 1024 + 8 * 16 * 1024 + 1024 * 1024  # 9 568 256 B


def _ops():
    """One entry per CUDA kernel of the forward pass.

    Each entry is (name, fa, fc, qa, qc, qw) with
        FLOPs = B * (fa * S^2 + fc)
        bytes = B * (qa * S^2 + qc) + qw
    """
    ops = []
    for i, (ci, co, k, din, dout) in enumerate(CONVS, 1):
        fa = 2 * ci * co * k * k / dout**2
        qa = F32 * (ci / din**2 + co / dout**2)
        ops.append((f"conv{i}", fa, 0, qa, 0, F32 * ci * co * k * k))
        ops.append((f"relu{i}", 0, 0, 2 * F32 * co / dout**2, 0, 0))
        if i == 1:
            # 32 x S/2 x S/2 -> 32 x S/4 x S/4, values (fp32) + indices (int64)
            ops.append(("maxpool", 0, 0, F32 * 32 / 4 + (F32 + I64) * 32 / 16, 0, 0))
    ops.append(("avgpool", 0, 0, F32 * 512 / 256, F32 * 512, 0))
    (i1, o1), (i2, o2) = FC
    ops.append(("fc1", 0, 2 * i1 * o1, 0, F32 * (i1 + o1), F32 * (i1 * o1 + o1)))
    ops.append(("relu_fc", 0, 0, 0, 2 * F32 * o1, 0))
    ops.append(("fc2", 0, 2 * i2 * o2, 0, F32 * (i2 + o2), F32 * (i2 * o2 + o2)))
    return ops


OPS = _ops()


def _sb(image_size, batch):
    S = np.asarray(image_size, dtype=float)
    B = np.asarray(batch, dtype=float)
    return S, B


def per_op(image_size, batch):
    """Lists of FLOPs and bytes per kernel, each broadcast over (S, B)."""
    S, B = _sb(image_size, batch)
    A = S * S
    fl = [B * (fa * A + fc) for _, fa, fc, _, _, _ in OPS]
    by = [B * (qa * A + qc) + qw for _, _, _, qa, qc, qw in OPS]
    return fl, by


def flops(image_size, batch):
    # = B * (17712 * S^2 + 313344)
    fl, _ = per_op(image_size, batch)
    return sum(fl)


def bytes_moved(image_size, batch):
    # = B * (380 * S^2 + 8592) + WEIGHT_BYTES
    _, by = per_op(image_size, batch)
    return sum(by)


def memory(image_size, batch):
    """Peak allocated bytes of one forward pass.

    The peak is inside the max-pool: input x (3S^2 floats) is still held by
    the caller, conv1 output (8S^2 floats) is the pool input, and the pool
    writes values (2S^2 floats) and int64 indices (2S^2).
        per image: 4*3 + 4*8 + 4*2 + 8*2 = 68 bytes per pixel of S^2
    Every later layer needs less (at most 36 S^2 B per image).
    = 13 729 552 + 68 * B * S^2
    """
    S, B = _sb(image_size, batch)
    return WEIGHT_BYTES + BLAS_WORKSPACE + 68.0 * B * S * S


def latency_terms(image_size, batch, theta):
    """Per-kernel time split into launch / memory / compute limited parts."""
    t_op, P, BW = theta["t_op"], theta["peak_flops"], theta["bandwidth"]
    fl, by = per_op(image_size, batch)
    launch = memory_t = compute = 0.0
    for f, q in zip(fl, by):
        tc, tm = f / P, q / BW
        t = np.maximum(t_op, np.maximum(tc, tm))
        launch = launch + np.where((t_op >= tc) & (t_op >= tm), t, 0.0)
        compute = compute + np.where((tc > t_op) & (tc >= tm), t, 0.0)
        memory_t = memory_t + np.where((tm > t_op) & (tm > tc), t, 0.0)
    return launch, memory_t, compute


def latency(image_size, batch, theta):
    return sum(latency_terms(image_size, batch, theta))


def energy(image_size, batch, theta_energy):
    """theta_energy holds p_static [W], e_flop [J/FLOP], e_byte [J/B] and
    the latency parameters under key 'latency'."""
    t = latency(image_size, batch, theta_energy["latency"])
    return (theta_energy["p_static"] * t
            + theta_energy["e_flop"] * flops(image_size, batch)
            + theta_energy["e_byte"] * bytes_moved(image_size, batch))


def regime(image_size, batch, theta):
    """0 = launch-bound, 1 = memory-bound, 2 = compute-bound (largest share)."""
    return np.argmax(np.stack(np.broadcast_arrays(*latency_terms(image_size, batch, theta))), axis=0)
