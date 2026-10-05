"""Stage 11: final TEST evaluation of the best layout checkpoint (run once)."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.data.layout_dataset import LayoutDataset, make_loader
from src.losses.layout_losses import box_iou_giou, build_constraints, relation_terms
from src.models.layout_model import LayoutModel
from src.models.text_encoder import get_embedder
from src.utils import get_device, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="layout")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

cfg = load_config(); lc = cfg["layout_loss"]; proc = cfg["paths"]["processed_dir"]
out_path = Path("outputs") / a.tag / "test_report.json"
if out_path.exists() and not a.force:
    sys.exit(f"{out_path} exists. The test set is meant to be evaluated once. Use --force to override.")

device = get_device()
emb = get_embedder(cfg, device)
ck = torch.load(Path("checkpoints") / a.tag / "best.pt", map_location=device, weights_only=False)
tc = ck["cfg"]
model = LayoutModel(emb.dim, tc["hidden"], tc["heads"], tc["layers"], tc["rel_dim"],
                    tc["mlp_hidden"], tc["dropout"]).to(device)
model.load_state_dict(ck["model"]); model.eval()
print(f"checkpoint epoch {ck['epoch']} | evaluating TEST split")

train_ds = LayoutDataset("train", proc)
mean_box = torch.cat([b for _, b in train_ds.items]).mean(0)
test_ds = LayoutDataset("test", proc)
loader = make_loader(test_ds, emb, 128, False)


class Acc:
    def __init__(self):
        self.iou, self.giou, self.l1 = [], [], []
        self.by_n = defaultdict(lambda: {"iou": [], "sat": 0, "tot": 0, "graphs": 0, "full": 0})
        self.by_depth = defaultdict(list)
        self.by_rel = defaultdict(lambda: [0, 0])      # [satisfied, total]

    def add(self, pred, gt, depth, cons):
        iou, giou = box_iou_giou(pred, gt)
        self.iou.append(iou); self.giou.append(giou); self.l1.append((pred - gt).abs().mean(-1))
        n = len(gt)
        g = self.by_n[n]
        g["iou"].append(iou); g["graphs"] += 1
        for d in range(n):
            self.by_depth[min(int(depth[d]), 3)].append(float(iou[d]))
        terms = relation_terms(pred, cons, lc)
        ok_all, has = True, False
        for r, v in terms.items():
            s = int((v <= 1e-9).sum())
            self.by_rel[r][0] += s; self.by_rel[r][1] += len(v)
            g["sat"] += s; g["tot"] += len(v)
            ok_all &= s == len(v); has = True
        if has:
            g["full"] += int(ok_all)

    def report(self):
        cat = lambda x: torch.cat(x)
        sat = sum(v[0] for v in self.by_rel.values()); tot = sum(v[1] for v in self.by_rel.values())
        return {
            "iou": float(cat(self.iou).mean()), "giou": float(cat(self.giou).mean()),
            "l1": float(cat(self.l1).mean()), "rel_sat": sat / tot,
            "by_n_objects": {n: {"graphs": g["graphs"], "iou": float(cat(g["iou"]).mean()),
                                 "rel_sat": g["sat"] / max(1, g["tot"]),
                                 "all_satisfied": g["full"] / max(1, g["graphs"])}
                             for n, g in sorted(self.by_n.items())},
            "by_depth_iou": {d: {"nodes": len(v), "iou": sum(v) / len(v)} for d, v in sorted(self.by_depth.items())},
            "by_relation_sat": {r: {"n": v[1], "sat": v[0] / v[1]} for r, v in sorted(self.by_rel.items())},
        }


accs = {"model": Acc(), "baseline": Acc()}
with torch.no_grad():
    for b in loader:
        bt = {k: v.to(device) for k, v in b["batch"].items()}
        pred = model(bt, b["plan"])["boxes"].float().cpu()
        gt, depth, off = b["gt"], b["plan"]["depth"], 0
        assert pred.shape == gt.shape
        for g in b["graphs"]:
            sl = slice(off, off + g.n); off += g.n
            cons = build_constraints([g])
            accs["model"].add(pred[sl], gt[sl], depth[sl], cons)
            accs["baseline"].add(mean_box.expand(g.n, 4).clone(), gt[sl], depth[sl], cons)

rep = {k: v.report() for k, v in accs.items()}
rep["checkpoint_epoch"] = ck["epoch"]; rep["n_test"] = len(test_ds)
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(out_path, "w"), indent=1)

M, B = rep["model"], rep["baseline"]
print(f"\n=== TEST ({len(test_ds)} graphs) model vs mean-box baseline ===")
for k in ("iou", "giou", "l1", "rel_sat"):
    print(f"{k:8s} model {M[k]:.3f} | baseline {B[k]:.3f}")
print("\n--- by number of objects (IoU | rel_sat | all-constraints-satisfied) ---")
for n, v in M["by_n_objects"].items():
    bv = B["by_n_objects"][n]
    print(f"n={n} graphs={v['graphs']:4d} model {v['iou']:.3f} {v['rel_sat']:.3f} {v['all_satisfied']:.3f}"
          f" | base {bv['iou']:.3f} {bv['rel_sat']:.3f} {bv['all_satisfied']:.3f}")
print("\n--- by BFS depth (node IoU; depth 0 = anchors) ---")
for d, v in M["by_depth_iou"].items():
    print(f"depth {d}{'+' if d == 3 else ' '} nodes={v['nodes']:5d} model {v['iou']:.3f} | base {B['by_depth_iou'][d]['iou']:.3f}")
print("\n--- by relation (satisfaction) ---")
for r, v in M["by_relation_sat"].items():
    print(f"{r:10s} n={v['n']:5d} model {v['sat']:.3f} | base {B['by_relation_sat'][r]['sat']:.3f}")
print("saved to", out_path)
