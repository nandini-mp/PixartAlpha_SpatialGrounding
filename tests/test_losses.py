"""Stage 9 tests: GIoU, constraint building, relation hinges, total loss, overfit."""
import zlib

import torch

from src.graph.batch import batch_graphs
from src.graph.graph import build_graph
from src.graph.plan import build_plan
from src.losses.layout_losses import (box_iou_giou, build_constraints, layout_loss,
                                      relation_loss, relation_terms)
from src.models.layout_model import LayoutModel
from src.parser.structure import parse_structure

CFG = {"w_l1": 1.0, "w_giou": 1.0, "w_rel": 0.5, "margin": 0.02, "near_dist": 0.35, "far_dist": 0.5}
TD = 16
CHAIN = "OBJECTS: 1 = A ; 2 = B ; 3 = C ; 4 = D RELATIONS: 1 above 2 ; 2 left 3 ; 3 below 4"
# GT boxes (cx,cy,w,h) satisfying CHAIN: A above B, B left C, C below D
GT = torch.tensor([[0.4, 0.2, 0.2, 0.2], [0.4, 0.5, 0.2, 0.3], [0.7, 0.5, 0.25, 0.2], [0.7, 0.2, 0.2, 0.25]])


def G(t):
    return build_graph(parse_structure(t)[0])


class FakeEmb:
    def get(self, descs):
        return torch.stack([torch.randn(TD, generator=torch.Generator().manual_seed(zlib.crc32(d.encode())))
                            for d in descs])


def T(*rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_giou_values():
    a, b = T([0.5, 0.5, 0.4, 0.4]), T([0.5, 0.5, 0.4, 0.4])
    iou, giou = box_iou_giou(a, b)
    assert torch.allclose(iou, torch.ones(1), atol=1e-5) and torch.allclose(giou, torch.ones(1), atol=1e-5)
    iou, giou = box_iou_giou(T([0.125, 0.25, 0.25, 0.5]), T([0.875, 0.75, 0.25, 0.5]))   # disjoint
    assert abs(float(iou)) < 1e-6 and abs(float(giou) + 0.75) < 1e-5
    iou, giou = box_iou_giou(T([0.25, 0.5, 0.5, 1.0]), T([0.5, 0.5, 0.5, 1.0]))          # half overlap
    assert abs(float(iou) - 1 / 3) < 1e-5 and abs(float(giou) - 1 / 3) < 1e-5


def test_constraints_chain():
    c = build_constraints([G(CHAIN)])
    assert c["pairs"]["LEFT"].tolist() == [[1, 2]]
    assert sorted(c["pairs"]["ABOVE"].tolist()) == [[0, 1], [3, 2]]      # 'C below D' -> D above C ... stored as (3,2)
    assert len(c["between"]) == 0


def test_constraints_offsets_symmetric_between():
    g1 = G("OBJECTS: 1 = a ; 2 = b RELATIONS: 1 near 2")
    g2 = G("OBJECTS: 1 = a ; 2 = b ; 3 = m RELATIONS: 3 between_y 1 2 ; 1 overlap 2")
    c = build_constraints([g1, g2])
    assert c["pairs"]["NEAR"].tolist() == [[0, 1]]                      # symmetric pair counted once
    assert c["pairs"]["OVERLAP"].tolist() == [[2, 3]]                   # offset +2
    assert c["between"].tolist() == [[4, 2, 3]] and c["between_axis"].tolist() == [1]


def test_gt_boxes_zero_loss_and_violation_positive():
    c = build_constraints([G(CHAIN)])
    assert float(relation_loss(GT, c, CFG)) == 0.0
    bad = GT.clone(); bad[0, 1] = 0.9                                    # A moved below B
    terms = relation_terms(bad, c, CFG)
    assert terms["ABOVE"].max() > 0.3


def test_each_relation_is_optimizable():
    cases = {"LEFT": "1 left 2", "ABOVE": "1 above 2", "INSIDE": "1 inside 2",
             "OVERLAP": "1 overlap 2", "NEAR": "1 near 2", "FAR": "1 far 2"}
    for name, rel in cases.items():
        c = build_constraints([G(f"OBJECTS: 1 = s ; 2 = d RELATIONS: {rel}")])
        start = T([0.5, 0.5, 0.2, 0.2], [0.55, 0.5, 0.2, 0.2]) if name == "FAR" else \
            T([0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3])
        x = start.clone().requires_grad_(True)
        assert float(relation_loss(x, c, CFG)) > 0, name
        opt = torch.optim.Adam([x], lr=0.02)
        for _ in range(500):
            opt.zero_grad(); relation_loss(x, c, CFG).backward(); opt.step()
        assert float(relation_loss(x, c, CFG)) < 1e-3, name
    for ax in "xy":
        c = build_constraints([G(f"OBJECTS: 1 = a ; 2 = b ; 3 = m RELATIONS: 3 between_{ax} 1 2")])
        x = T([0.1, 0.1, 0.2, 0.2], [0.9, 0.9, 0.2, 0.2], [0.95, 0.05, 0.2, 0.2]).requires_grad_(True)
        assert float(relation_loss(x, c, CFG)) > 0
        opt = torch.optim.Adam([x], lr=0.02)
        for _ in range(500):
            opt.zero_grad(); relation_loss(x, c, CFG).backward(); opt.step()
        assert float(relation_loss(x, c, CFG)) < 1e-3, ax


def test_empty_constraints():
    c = build_constraints([G("OBJECTS: 1 = a ; 2 = b RELATIONS: none")])
    x = GT[:2].clone().requires_grad_(True)
    l = relation_loss(x, c, CFG)
    assert float(l) == 0.0
    l.backward()


def test_total_loss_perfect_and_backward():
    c = build_constraints([G(CHAIN)])
    out = layout_loss(GT.clone(), GT, c, CFG)
    assert float(out["total"]) < 1e-5 and abs(float(out["iou"]) - 1) < 1e-4
    x = (GT + 0.05).requires_grad_(True)
    out = layout_loss(x, GT, c, CFG)
    out["total"].backward()
    assert out["total"] > 0 and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_overfit_single_layout():
    torch.manual_seed(0)
    gs = [G(CHAIN)]
    batch, plan, c = batch_graphs(gs, FakeEmb()), build_plan(gs), build_constraints(gs)
    m = LayoutModel(TD, 32, 4, 2, rel_dim=8, mlp_hidden=64).train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for step in range(300):
        opt.zero_grad()
        loss = layout_loss(m(batch, plan)["boxes"], GT, c, CFG)["total"]
        if first is None:
            first = float(loss)
        loss.backward(); opt.step()
    assert float(loss) < 0.35 * first, (first, float(loss))
