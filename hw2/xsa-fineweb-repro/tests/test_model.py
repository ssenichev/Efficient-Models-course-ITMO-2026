import pytest
import torch
from xsa_repro.model import GPT, ModelConfig


def test_main_count_and_pair():
    # Meta device verifies the real module structure without allocating 350 MB.
    with torch.device("meta"):
        sa, xsa = GPT(ModelConfig(), "sa"), GPT(ModelConfig(), "xsa")
    assert sa.parameter_count() == xsa.parameter_count() == 44_647_936
    assert sa.wte.weight is sa.lm_head.weight


def test_initialization_and_shapes():
    c = ModelConfig(vocab_size=64, n_layer=2, n_embd=32, n_head=4, block_size=8)
    torch.manual_seed(42)
    sa = GPT(c, "sa")
    torch.manual_seed(42)
    xsa = GPT(c, "xsa")
    assert all(torch.equal(v, xsa.state_dict()[k]) for k, v in sa.state_dict().items())
    ids = torch.randint(0, 64, (2, 8))
    for model in (sa, xsa):
        logits, loss = model(ids, ids)
        assert logits.shape == (2, 8, 64)
        assert loss.ndim == 0
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("kind", ["sa", "xsa"])
def test_causality(kind):
    model = GPT(ModelConfig(vocab_size=32, n_embd=16, n_layer=2, n_head=2, block_size=8), kind).eval()
    a = torch.randint(0, 32, (2, 8))
    b = a.clone()
    b[:, 4:] = (b[:, 4:] + 1) % 32
    with torch.no_grad():
        torch.testing.assert_close(model(a)[0][:, :4], model(b)[0][:, :4], atol=1e-6, rtol=1e-6)
