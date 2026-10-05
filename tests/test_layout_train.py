"""Stage 10 tests: INSIDE slack, teacher-forcing schedule, collate, evaluation."""
import zlib

import torch

from src.data.layout_dataset import make_collate
from src.evaluation.layout_metrics import evaluate
from src.graph.graph import build_graph
from src.losses.layout_losses import build_constraints, relation_terms
from src.parser.structure import parse_structure
from src.training.layout_trainer import tf_ratio

CFG = {"w_l1": 1.0, "w_giou": 1.0, "w_rel": 0.5, "margin": 0.02, "near_dist": 0.35, "far_dist": 0.5}
TD = 16
CHAIN = "OBJECTS: 1 = A ; 2 = B ; 3 = C ; 4 = D RELATIONS: 1 above 2 ; 2 left 3 ; 3 below 4"
PAIR = "OBJECTS: 1 = a ; 2 = b RELATIONS: 1 above 2"
GT = torch.tensor([[0.4, 0.2, 0.2, 0.2], [0.4, 0.5, 0.2, 0.3], [0.7, 0.5, 0.25, 0.2], [0.7, 0.2, 0.2, 0.25]])


def G(t):
    return build_graph(parse_structure(t)[0])


class FakeEmb:
    def get(self, descs):
        return torch.stack([torch.randn(TD, generator=torch.Generator().manual_seed(zlib.crc32(d.encode())))
                            for d in descs])


def items():
    return [(G(CHAIN), GT.clone()), (G(PAIR), GT[:2].clone())]


def test_inside_slack_tolerates_small_violation():
    c = build_constraints([G("OBJECTS: 1 = s ; 2 = d RELATIONS: 1 inside 2")])
    boxes = torch.tensor([[0.5, 0.5, 0.42, 0.42], [0.5, 0.5, 0.4, 0.4]])      # s sticks out by 0.01 per side
    assert relation_terms(boxes, c, CFG)["INSIDE"].max() > 0
    assert relation_terms(boxes, c, {**CFG, "inside_slack": 0.03})["INSIDE"].max() == 0


def test_tf_schedule():
    tc = {"tf_start": 0.5, "tf_end": 0.0, "tf_decay_epochs": 10}
    assert tf_ratio(0, tc) == 0.5 and abs(tf_ratio(5, tc) - 0.25) < 1e-9
    assert tf_ratio(10, tc) == 0.0 and tf_ratio(50, tc) == 0.0


def test_collate_shapes():
    b = make_collate(FakeEmb())(items())
    assert b["gt"].shape == (6, 4) and b["batch"]["x_text"].shape == (6, TD)
    assert b["plan"]["depth"].shape[0] == 6 and len(b["graphs"]) == 2
    assert b["cons"]["pairs"]["LEFT"].shape[0] == 1 and b["cons"]["pairs"]["ABOVE"].shape[0] == 3


def test_evaluate_perfect_and_mean_box():
    loader = [make_collate(FakeEmb())(items())]
    r = evaluate(lambda b, dev: b["gt"].to(dev), loader, CFG, "cpu")
    assert abs(r["iou"] - 1) < 1e-4 and r["l1"] < 1e-6 and r["rel_sat"] == 1.0
    mb = torch.tensor([0.5, 0.5, 0.3, 0.4])
    base = evaluate(lambda b, dev: mb.expand(len(b["gt"]), 4).clone(), loader, CFG, "cpu")
    assert base["iou"] < 0.5 and base["rel_sat"] < 1.0 and "ABOVE" in base["per_rel"]
