"""Stage 7 tests: box decode/encode, anchor head, non-anchor head."""
import math

import torch
import torch.nn as nn

from src.graph.relations import REL2ID
from src.models.boxes import decode_box, encode_delta, MIN_SIZE
from src.models.layout_heads import AnchorMLP, NonAnchorMLP

H = 32


def strong(head):
    """Re-init the last layer so the output is clearly input-dependent (the default init is ~0)."""
    nn.init.normal_(head.net[-1].weight, std=0.5)
    return head


def rand_inputs(B, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return (torch.randint(0, 6, (B,), generator=g), torch.rand(B, 2, generator=g), r(B, H),
            torch.rand(B, 4, generator=g) * 0.5 + 0.2, r(B, H))


def test_anchor_output_range():
    torch.manual_seed(0)
    m = strong(AnchorMLP(H)).eval()
    b = m(torch.randn(200, H) * 5)
    assert b.shape == (200, 4) and torch.isfinite(b).all()
    assert (b[:, :2] >= 0).all() and (b[:, :2] <= 1).all()
    assert (b[:, 2:] >= MIN_SIZE - 1e-6).all() and (b[:, 2:] <= 1).all()


def test_anchor_init_is_centered_medium_box():
    torch.manual_seed(0)
    b = AnchorMLP(H).eval()(torch.randn(50, H))
    assert (b[:, :2] - 0.5).abs().max() < 0.05
    assert (b[:, 2:] - 0.3).abs().max() < 0.05


def test_decode_zero_delta_returns_parent():
    p = torch.tensor([[0.4, 0.6, 0.3, 0.2]])
    assert torch.allclose(decode_box(p, torch.zeros(1, 4)), p, atol=1e-6)


def test_decode_formula():
    p = torch.tensor([[0.5, 0.5, 0.4, 0.2]])
    d = torch.tensor([[0.5, -1.0, math.log(2.0), math.log(0.5)]])
    out = decode_box(p, d)
    assert torch.allclose(out, torch.tensor([[0.7, 0.3, 0.8, 0.1]]), atol=1e-6)


def test_encode_decode_roundtrip():
    torch.manual_seed(1)
    parent = torch.cat([torch.rand(500, 2) * 0.6 + 0.2, torch.rand(500, 2) * 0.5 + 0.3], -1)
    child = torch.cat([torch.rand(500, 2), torch.rand(500, 2) * 0.5 + 0.1], -1)
    rec = decode_box(parent, encode_delta(parent, child))
    assert torch.allclose(rec, child, atol=1e-5)


def test_decode_clamps_and_stays_finite():
    p = torch.tensor([[0.9, 0.9, 0.8, 0.8]])
    out = decode_box(p, torch.tensor([[50.0, -50.0, 100.0, -100.0]]))
    assert torch.isfinite(out).all()
    assert (out[:, :2] >= 0).all() and (out[:, :2] <= 1).all()
    assert out[0, 2] <= 1.0 and out[0, 3] >= MIN_SIZE


def test_nonanchor_init_child_equals_parent():
    torch.manual_seed(2)
    m = NonAnchorMLP(H).eval()
    rel, ac, nf, pb, pf = rand_inputs(64)
    delta, box = m(rel, ac, nf, pb, pf)
    assert delta.shape == (64, 4) and box.shape == (64, 4)
    assert delta.abs().max() < 0.1
    assert (box - pb).abs().max() < 0.1


def test_nonanchor_all_inputs_matter_and_get_gradients():
    torch.manual_seed(3)
    m = strong(NonAnchorMLP(H)).eval()
    rel, ac, nf, pb, pf = rand_inputs(16)
    ac, nf, pb, pf = [t.clone().requires_grad_(True) for t in (ac, nf, pb, pf)]
    delta, box = m(rel, ac, nf, pb, pf)
    (delta * torch.randn_like(delta)).sum().backward()
    for name, t in [("anchor_c", ac), ("node_feat", nf), ("parent_box", pb), ("parent_feat", pf)]:
        assert t.grad is not None and t.grad.abs().sum() > 0, name
    assert m.rel_emb.weight.grad.abs().sum() > 0
    base = m(rel, ac, nf, pb, pf)[0]
    for k, alt in enumerate([(rel, ac + 0.3, nf, pb, pf), (rel, ac, nf + 1, pb, pf),
                             (rel, ac, nf, pb + 0.1, pf), (rel, ac, nf, pb, pf + 1)]):
        assert not torch.allclose(base, m(*alt)[0], atol=1e-6), k


def test_relation_changes_output():
    torch.manual_seed(4)
    m = strong(NonAnchorMLP(H)).eval()
    _, ac, nf, pb, pf = rand_inputs(8)
    l = torch.full((8,), REL2ID["LEFT"]); r = torch.full((8,), REL2ID["RIGHT"])
    assert not torch.allclose(m(l, ac, nf, pb, pf)[0], m(r, ac, nf, pb, pf)[0], atol=1e-6)


def test_batch_independence():
    torch.manual_seed(5)
    m = strong(NonAnchorMLP(H)).eval()
    inp = rand_inputs(6)
    _, full = m(*inp)
    _, one = m(*[t[2:3] for t in inp])
    assert torch.allclose(full[2:3], one, atol=1e-5)


def test_input_dim():
    assert NonAnchorMLP(H, rel_dim=64).in_dim == 64 + 2 + H + 4 + H
