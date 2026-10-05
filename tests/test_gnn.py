"""Stage 6 tests: scatter-softmax, edge-awareness, equivariance, batching, receptive field, grads."""
import zlib
from pathlib import Path

import pytest
import torch

from src.graph.batch import batch_graphs
from src.graph.graph import build_graph
from src.graph.relations import REL2ID
from src.models.gnn import EdgeAwareGNN, scatter_softmax
from src.parser.structure import parse_structure
from src.utils import load_config

TD, HID = 16, 32


def net(layers=3, seed=0):
    torch.manual_seed(seed)
    return EdgeAwareGNN(TD, HID, heads=4, layers=layers).eval()


def both(pairs, types):
    """pairs [(s,d)...] with type list -> edge_index/edge_type containing both directions given."""
    return torch.tensor(pairs).t().contiguous(), torch.tensor([REL2ID[t] for t in types])


def chain_edges(n, fwd="LEFT", bwd="RIGHT"):
    pairs, types = [], []
    for i in range(n - 1):
        pairs += [(i, i + 1), (i + 1, i)]
        types += [fwd, bwd]
    return both(pairs, types)


def test_scatter_softmax_matches_dense():
    torch.manual_seed(0)
    logits, idx = torch.randn(7, 2), torch.tensor([0, 0, 1, 1, 1, 3, 3])
    out = scatter_softmax(logits, idx, 4)
    for g in (0, 1, 3):
        m = idx == g
        assert torch.allclose(out[m], torch.softmax(logits[m], dim=0), atol=1e-6)
        assert torch.allclose(out[m].sum(0), torch.ones(2), atol=1e-6)


def test_relation_type_changes_output():
    torch.manual_seed(1); x = torch.randn(2, TD)
    ei, e1 = both([(0, 1), (1, 0)], ["LEFT", "RIGHT"])
    _, e2 = both([(0, 1), (1, 0)], ["ABOVE", "BELOW"])
    _, e3 = both([(0, 1), (1, 0)], ["RIGHT", "LEFT"])          # opposite direction of the same pair
    m = net()
    a, b, c = m(x, ei, e1), m(x, ei, e2), m(x, ei, e3)
    assert not torch.allclose(a, b, atol=1e-5) and not torch.allclose(a, c, atol=1e-5)


def test_neighbor_features_change_output():
    torch.manual_seed(2); x = torch.randn(2, TD); x2 = x.clone(); x2[1] = torch.randn(TD)
    ei, et = both([(0, 1), (1, 0)], ["LEFT", "RIGHT"])
    m = net()
    assert not torch.allclose(m(x, ei, et)[0], m(x2, ei, et)[0], atol=1e-5)


def test_permutation_equivariance():
    torch.manual_seed(3); n = 6; x = torch.randn(n, TD)
    ei, et = both([(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 1), (1, 4)],
                  ["LEFT", "RIGHT", "ABOVE", "BELOW", "NEAR", "NEAR", "INSIDE", "CONTAINS"])
    m = net(); out = m(x, ei, et)
    p = torch.randperm(n); xn = torch.empty_like(x); xn[p] = x
    out_n = m(xn, p[ei], et)
    assert torch.allclose(out_n[p], out, atol=1e-5)


def test_isolated_nodes_and_no_edges():
    x = torch.randn(3, TD)
    out = net()(x, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert out.shape == (3, HID) and torch.isfinite(out).all()


def test_batch_equals_individual():
    torch.manual_seed(4); xa, xb = torch.randn(3, TD), torch.randn(2, TD)
    ea, ta = chain_edges(3); eb, tb = chain_edges(2, "ABOVE", "BELOW")
    m = net()
    ya, yb = m(xa, ea, ta), m(xb, eb, tb)
    y = m(torch.cat([xa, xb]), torch.cat([ea, eb + 3], 1), torch.cat([ta, tb]))
    assert torch.allclose(y[:3], ya, atol=1e-5) and torch.allclose(y[3:], yb, atol=1e-5)


def test_receptive_field_is_multi_hop():
    torch.manual_seed(5); x = torch.randn(4, TD); x2 = x.clone(); x2[0] = torch.randn(TD)     # perturb node 0
    ei, et = chain_edges(4)                                                      # 0-1-2-3
    for layers, should_change in ((3, True), (2, False)):
        m = net(layers)
        changed = not torch.allclose(m(x, ei, et)[3], m(x2, ei, et)[3], atol=1e-6)
        assert changed == should_change, (layers, changed)


def test_gradients_reach_relation_embedding_and_node_mlp():
    torch.manual_seed(6); x = torch.randn(3, TD); ei, et = chain_edges(3)
    m = net(); out = m(x, ei, et)
    (out * torch.randn_like(out)).sum().backward()
    g = m.rel_emb.weight.grad
    assert g[REL2ID["LEFT"]].abs().sum() > 0 and g[REL2ID["RIGHT"]].abs().sum() > 0
    assert g[REL2ID["FAR"]].abs().sum() == 0                                      # unused relation
    assert all(p.grad is not None for p in m.node_in.parameters())


class FakeEmb:
    def get(self, descs):
        return torch.stack([torch.randn(TD, generator=torch.Generator().manual_seed(zlib.crc32(d.encode())))
                            for d in descs])


def test_batch_graphs_offsets_and_duplicates():
    def g(t): return build_graph(parse_structure(t)[0])
    g1 = g("OBJECTS: 1 = person ; 2 = person ; 3 = dog RELATIONS: 1 left 2 ; 3 near 1")
    g2 = g("OBJECTS: 1 = cat ; 2 = cup RELATIONS: 1 above 2")
    b = batch_graphs([g1, g2], FakeEmb())
    assert b["x_text"].shape == (5, TD) and b["offsets"].tolist() == [0, 3]
    assert b["node_graph"].tolist() == [0, 0, 0, 1, 1]
    assert b["edge_index"].shape[1] == len(g1.edges) + len(g2.edges) == b["edge_type"].shape[0]
    assert b["edge_index"][:, len(g1.edges):].min() >= 3                           # offset applied
    assert not torch.allclose(b["x_text"][0], b["x_text"][1]) is False             # duplicates: same text...
    assert torch.allclose(b["x_text"][0], b["x_text"][1])                          # ...same embedding, 2 nodes
    m = EdgeAwareGNN(TD, HID, 4, 3).eval()
    y = m(b["x_text"], b["edge_index"], b["edge_type"])
    assert not torch.allclose(y[0], y[1], atol=1e-5)                               # relations separate them


def test_cached_clip_embeddings():
    p = Path(load_config()["paths"]["processed_dir"]) / "desc_embeddings.pt"
    if not p.exists():
        pytest.skip("run scripts/embed_descriptions.py first")
    emb = torch.load(p, weights_only=True)["emb"]
    assert "small dog" in emb or "dog" in emb
    for v in emb.values():
        assert v.shape == (512,) and abs(float(v.norm()) - 1) < 1e-3
    if {"dog", "small dog", "car"} <= set(emb):
        assert float(emb["dog"] @ emb["small dog"]) > float(emb["dog"] @ emb["car"])
