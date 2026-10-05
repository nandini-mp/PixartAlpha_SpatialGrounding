"""Stage 8 tests: plan tensors, traversal order, teacher forcing, multi-hop gradients, batching."""
import zlib

import torch

from src.graph.batch import batch_graphs
from src.graph.graph import build_graph
from src.graph.plan import build_plan
from src.graph.relations import REL2ID
from src.models.boxes import decode_box, MIN_SIZE
from src.models.layout_model import LayoutModel
from src.parser.structure import parse_structure

TD, HID = 16, 32
CHAIN = "OBJECTS: 1 = A ; 2 = B ; 3 = C ; 4 = D RELATIONS: 1 above 2 ; 2 left 3 ; 3 below 4"
MIXED = "OBJECTS: 1 = a ; 2 = b ; 3 = c ; 4 = d ; 5 = e RELATIONS: 1 above 2 ; 4 right 5"
ONE = "OBJECTS: 1 = dog RELATIONS: none"


class FakeEmb:
    def get(self, descs):
        return torch.stack([torch.randn(TD, generator=torch.Generator().manual_seed(zlib.crc32(d.encode())))
                            for d in descs])


def G(t):
    return build_graph(parse_structure(t)[0])


def make_model(seed=0, train=False):
    torch.manual_seed(seed)
    m = LayoutModel(TD, HID, 4, 2, rel_dim=8, mlp_hidden=32)
    torch.nn.init.normal_(m.nonanchor.net[-1].weight, std=0.3)      # make outputs input-dependent
    torch.nn.init.normal_(m.anchor.net[-1].weight, std=0.3)
    return m.train() if train else m.eval()


def run(m, graphs, gt=None, tf=0.0, seed=0):
    return m(batch_graphs(graphs, FakeEmb()), build_plan(graphs), gt, tf, torch.Generator().manual_seed(seed))


def test_plan_chain():
    p = build_plan([G(CHAIN)])
    assert p["depth"].tolist() == [1, 0, 1, 2]
    assert p["parent"].tolist() == [1, -1, 1, 2]
    assert p["anchor_of"].tolist() == [1, 1, 1, 1]
    assert p["rel"].tolist() == [REL2ID["ABOVE"], -1, REL2ID["RIGHT"], REL2ID["ABOVE"]]


def test_plan_disconnected_and_batched_offsets():
    g1, g2 = G(MIXED), G(CHAIN)
    p = build_plan([g1, g2])
    assert p["depth"][:5].tolist() == [0, 1, 0, 0, 1]                # 3 components -> 3 anchors
    assert p["anchor_of"][:5].tolist() == [0, 0, 2, 3, 3]
    assert p["parent"][5:].tolist() == [6, -1, 6, 7]                 # offsets (+5) applied
    assert int((p["depth"] == 0).sum()) == len(g1.components) + len(g2.components)


def test_shapes_validity_and_anchor_box():
    out = run(make_model(), [G(MIXED), G(CHAIN), G(ONE)])
    b = out["boxes"]
    assert b.shape == (10, 4) and torch.isfinite(b).all() and torch.isfinite(out["deltas"]).all()
    assert (b[:, :2] >= 0).all() and (b[:, :2] <= 1).all()
    assert (b[:, 2:] >= MIN_SIZE - 1e-6).all() and (b[:, 2:] <= 1).all()
    assert out["deltas"][[0, 2, 3, 6, 9]].abs().sum() == 0           # anchors have no delta


def test_children_decoded_from_predicted_parents():
    gs = [G(MIXED), G(CHAIN)]
    out = run(make_model(), gs)
    p = build_plan(gs)
    ch = (p["depth"] > 0).nonzero(as_tuple=True)[0]
    exp = decode_box(out["boxes"][p["parent"][ch]], out["deltas"][ch])
    assert torch.allclose(out["boxes"][ch], exp, atol=1e-6)


def test_teacher_forcing_uses_gt_parent_boxes():
    gs = [G(CHAIN)]
    torch.manual_seed(7); gt = torch.rand(4, 4) * 0.5 + 0.2
    out = run(make_model(train=True), gs, gt, tf=1.0)
    p = build_plan(gs)
    ch = (p["depth"] > 0).nonzero(as_tuple=True)[0]
    exp = decode_box(gt[p["parent"][ch]], out["deltas"][ch])
    assert torch.allclose(out["boxes"][ch], exp, atol=1e-6)


def test_teacher_forcing_ignored_in_eval_mode():
    gs = [G(CHAIN)]
    gt = torch.rand(4, 4) * 0.5 + 0.2
    m = make_model()
    assert torch.allclose(run(m, gs, gt, tf=1.0)["boxes"], run(m, gs, None, tf=0.0)["boxes"])


def test_gradient_flows_through_multihop_chain_to_anchor_head():
    m = make_model(train=True)
    out = run(m, [G(CHAIN)], tf=0.0)
    out["boxes"][3].sum().backward()                                 # node 3 is at depth 2
    assert m.anchor.net[0].weight.grad.abs().sum() > 0               # anchor head influences it
    assert m.nonanchor.rel_emb.weight.grad.abs().sum() > 0
    assert m.gnn.rel_emb.weight.grad.abs().sum() > 0


def test_full_teacher_forcing_cuts_gradient_to_anchor_head():
    m = make_model(train=True)
    gt = torch.rand(4, 4) * 0.5 + 0.2
    out = run(m, [G(CHAIN)], gt, tf=1.0)
    out["boxes"][[0, 2, 3]].sum().backward()                         # non-anchor nodes only
    for prm in m.anchor.parameters():
        assert prm.grad is None or prm.grad.abs().sum() == 0
    assert m.nonanchor.net[0].weight.grad.abs().sum() > 0


def test_batch_equals_individual():
    m = make_model(1)
    g1, g2 = G(CHAIN), G(MIXED)
    both = run(m, [g1, g2])["boxes"]
    assert torch.allclose(both[:4], run(m, [g1])["boxes"], atol=1e-5)
    assert torch.allclose(both[4:], run(m, [g2])["boxes"], atol=1e-5)


def test_single_node_and_no_edges():
    out = run(make_model(), [G(ONE), G(ONE)])
    assert out["boxes"].shape == (2, 4) and torch.isfinite(out["boxes"]).all()
