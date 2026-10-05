"""Mean-box baseline on val + how often GT boxes satisfy the hinge definitions."""
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.graph.graph import build_graph
from src.losses.layout_losses import box_iou_giou, build_constraints, relation_terms
from src.utils import load_config

cfg = load_config(); lc = cfg["layout_loss"]
proc = Path(cfg["paths"]["processed_dir"])


def load(split):
    out = []
    for s in json.load(open(proc / f"{split}_prompts.json")):
        st = {"objects": [{"id": o["id"], "desc": o["desc"]} for o in s["struct"]["objects"]],
              "relations": s["struct"]["relations"]}
        out.append((build_graph(st), torch.tensor(s["boxes"], dtype=torch.float32)))
    return out


train, val = load("train"), load("val")
mean_box = torch.cat([b for _, b in train]).mean(0)
print("train mean box (cx,cy,w,h):", [round(float(x), 3) for x in mean_box])


def stats(data, boxes_fn):
    viol, tot, hinge = defaultdict(int), defaultdict(int), defaultdict(float)
    ious, l1s = [], []
    for g, gt in data:
        pred = boxes_fn(gt)
        iou, _ = box_iou_giou(pred, gt)
        ious.append(iou); l1s.append((pred - gt).abs().mean(-1))
        for r, v in relation_terms(pred, build_constraints([g]), lc).items():
            viol[r] += int((v > 1e-9).sum()); tot[r] += len(v); hinge[r] += float(v.sum())
    return torch.cat(ious).mean().item(), torch.cat(l1s).mean().item(), viol, tot, hinge


for name, fn in [("GT boxes (definition check)", lambda gt: gt),
                 ("mean-box baseline", lambda gt: mean_box.expand_as(gt).clone())]:
    iou, l1, viol, tot, hinge = stats(val, fn)
    print(f"\n=== {name} | val: mean IoU {iou:.3f} | mean L1 {l1:.3f} ===")
    for r in sorted(tot):
        print(f"  {r:10s} n={tot[r]:5d} violated={viol[r] / tot[r] * 100:6.2f}% mean_hinge={hinge[r] / tot[r]:.4f}")
    allv, allt = sum(viol.values()), sum(tot.values())
    print(f"  ALL        n={allt} violated={allv / allt * 100:.2f}%")
