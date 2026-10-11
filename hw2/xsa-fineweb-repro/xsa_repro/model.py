"""Original NanoGPT-style implementation of SA and per-head XSA."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_embd: int = 512
    n_head: int = 8
    block_size: int = 1024
    mlp_ratio: int = 4
    rope_theta: float = 10000.0
    embed_layer_norm: bool = True
    tie_embeddings: bool = True
    bias: bool = True
    dropout: float = 0.0
    xsa_projection_fp32: bool = False

    def __post_init__(self):
        if min(self.vocab_size, self.n_layer, self.n_embd, self.n_head, self.block_size, self.mlp_ratio) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("n_embd must divide into heads with even RoPE head_dim")
        if self.dropout != 0.0:
            raise ValueError("Controlled experiments require dropout=0")
        if self.rope_theta <= 0:
            raise ValueError("RoPE theta must be positive")


def apply_rope(q, k, theta=10000.0):
    """Rotate adjacent pairs of Q/K, leaving values outside this API.

    Frequencies use fp32 for mixed precision, fp64 for double reference tests.
    """
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    d, t = q.size(-1), q.size(-2)
    freq = theta ** (-torch.arange(0, d, 2, device=q.device, dtype=dtype) / d)
    angles = torch.arange(t, device=q.device, dtype=dtype)[:, None] * freq[None, :]
    c, s = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

    def rotate(x):
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


def exclusive_projection(y, v, fp32=False, eps=1e-6):
    original_dtype = y.dtype
    if fp32 and y.dtype in (torch.float16, torch.bfloat16):
        y, v = y.float(), v.float()
    vn = F.normalize(v, p=2, dim=-1, eps=eps)
    return (y - (y * vn).sum(dim=-1, keepdim=True) * vn).to(original_dtype)


def attention(q, k, v, attention_type="sa", projection_fp32=False):
    if attention_type not in ("sa", "xsa"):
        raise ValueError("attention_type must be sa or xsa")
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    z = exclusive_projection(y, v, projection_fp32) if attention_type == "xsa" else y
    return y, z


class CausalSelfAttention(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        self.config, self.attention_type = config, attention_type
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Diagnostics consume detached tensors immediately; ordinary training stores none.
        self.diagnostic_hook = None

    def qkv(self, x):
        b, t, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=-1)
        q, k, v = [a.view(b, t, self.config.n_head, d // self.config.n_head).transpose(1, 2)
                   for a in (q, k, v)]
        q, k = apply_rope(q, k, self.config.rope_theta)
        return q, k, v

    def forward(self, x):
        q, k, v = self.qkv(x)
        y, z = attention(q, k, v, self.attention_type, self.config.xsa_projection_fp32)
        if self.diagnostic_hook is not None:
            self.diagnostic_hook(y.detach(), z.detach(), v.detach())
        return self.c_proj(z.transpose(1, 2).contiguous().view_as(x))


class Block(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, attention_type)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.ModuleDict({
            "c_fc": nn.Linear(config.n_embd, config.mlp_ratio * config.n_embd, bias=config.bias),
            "c_proj": nn.Linear(config.mlp_ratio * config.n_embd, config.n_embd, bias=config.bias),
        })

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp["c_proj"](F.gelu(self.mlp["c_fc"](self.ln_2(x))))


class GPT(nn.Module):
    def __init__(self, config, attention_type="sa"):
        super().__init__()
        if attention_type not in ("sa", "xsa"):
            raise ValueError("attention_type must be sa or xsa")
        self.config, self.attention_type = config, attention_type
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.embed_norm = nn.LayerNorm(config.n_embd, bias=config.bias) if config.embed_layer_norm else nn.Identity()
        self.blocks = nn.ModuleList([Block(config, attention_type) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        # NanoGPT output head has no bias; Linear/LayerNorm in blocks have bias.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 0 < idx.size(1) <= self.config.block_size:
            raise ValueError("Expected [batch, time] with 0 < time <= block_size")
        x = self.embed_norm(self.wte(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
