import math
import pytest
import torch
from torch.nn import functional as F
from xsa_repro.model import attention, exclusive_projection, apply_rope, CausalSelfAttention, ModelConfig


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_independent_reference(dtype):
    torch.manual_seed(7)
    q, k, v = [torch.randn(2, 3, 7, 8, dtype=dtype) for _ in range(3)]
    scores = q @ k.transpose(-2, -1) / math.sqrt(8)
    scores.masked_fill_(torch.ones(7, 7, dtype=torch.bool).triu(1), float("-inf"))
    ref_y = scores.softmax(-1) @ v
    ref_z = ref_y - ((ref_y * v).sum(-1, keepdim=True) / v.square().sum(-1, keepdim=True)) * v
    y, z = attention(q, k, v, "xsa")
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(y, ref_y, atol=tol, rtol=tol)
    torch.testing.assert_close(z, ref_z, atol=tol, rtol=tol)
    sy, sz = attention(q, k, v, "sa")
    assert sy is sz
    torch.testing.assert_close(sy, F.scaled_dot_product_attention(q, k, v, is_causal=True))


@pytest.mark.parametrize("dtype,fp32,tol", [(torch.float32, False, 2e-6), (torch.bfloat16, False, 0.04), (torch.bfloat16, True, 0.01)])
def test_orthogonality(dtype, fp32, tol):
    torch.manual_seed(6)
    y, v = [torch.randn(2, 3, 6, 8).to(dtype) for _ in range(2)]
    z = exclusive_projection(y, v, fp32)
    cosine = F.cosine_similarity(z.float(), v.float(), dim=-1)
    assert cosine.abs().max() < tol
    assert z.dtype == dtype


def test_single_token():
    q, k, v = [torch.randn(2, 4, 1, 8) for _ in range(3)]
    _, z = attention(q, k, v, "xsa")
    assert z.abs().max() < 1e-6


def test_zero_value_leaves_arbitrary_output_unchanged():
    y = torch.randn(2, 3, 4, 8)
    assert torch.equal(exclusive_projection(y, torch.zeros_like(y)), y)


@pytest.mark.parametrize("fp32", [False, True])
def test_bf16_backward(fp32):
    y, v = [torch.randn(1, 2, 4, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    exclusive_projection(y, v, fp32).float().square().sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in (y, v))


@pytest.mark.parametrize("scale", [0.0, 1e-12, 1.0])
def test_finite_backward(scale):
    q, k = [torch.randn(1, 2, 4, 8, requires_grad=True) for _ in range(2)]
    v = (torch.randn(1, 2, 4, 8) * scale).requires_grad_()
    _, z = attention(q, k, v, "xsa")
    assert torch.isfinite(z).all()
    if scale == 0:
        assert torch.count_nonzero(z) == 0
    z.square().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))


def test_projection_value_gradient_not_detached():
    y = torch.randn(1, 1, 3, 4, dtype=torch.double, requires_grad=True)
    v = torch.randn_like(y, requires_grad=True)
    assert torch.autograd.gradcheck(exclusive_projection, (y, v))


def test_rope_values_and_output_projection():
    c = ModelConfig(vocab_size=32, n_embd=16, n_head=2, n_layer=1, block_size=8)
    m = CausalSelfAttention(c, "xsa")
    x = torch.randn(2, 5, 16)
    raw_q, raw_k, raw_v = [a.view(2, 5, 2, 8).transpose(1, 2) for a in m.c_attn(x).split(16, -1)]
    q, k, v = m.qkv(x)
    assert torch.equal(v, raw_v)
    assert not torch.equal(q[:, :, 1:], raw_q[:, :, 1:])
    rq, rk = apply_rope(raw_q, raw_k)
    torch.testing.assert_close(q, rq)
    torch.testing.assert_close(k, rk)
    _, z = attention(q, k, v, "xsa")
    torch.testing.assert_close(m(x), m.c_proj(z.transpose(1, 2).reshape(2, 5, 16)))
