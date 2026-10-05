"""Stage 12: figures for best / worst / random / multi-hop TEST graphs (uses the saved checkpoint)."""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.data.layout_dataset import LayoutDataset, make_collate
from src.graph.batch import batch_graphs
from src.graph.plan import build_plan
from src.losses.layout_losses import box_iou_giou, build_constraints, relation_terms
from src.models.layout_model import LayoutModel
from src.models.text_encoder import get_embedder
from src.utils import get_device, load_config
from src.visualization.layout_plot import plot_pair

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="layout")
ap.add_argument("--split", default="test")
ap.add_argument("--k", type=int, default=6)
a = ap.parse_args()

cfg = load_config(); lc = cfg["layout_loss"]
device = get_device(); emb = get_embedder(cfg, device)
ck = torch.load(Path("checkpoints") / a.tag / "best.pt", map_location=device, weights_only=False)
tc = ck["cfg"]
model = LayoutModel(emb.dim, tc["hidden"], tc["heads"], tc["layers"], tc["rel_dim"],
                    tc["mlp_hidden"], tc["dropout"]).to(device)
model.load_state_dict(ck["model"]); model.eval()

ds = LayoutDataset(a.split, cfg["paths"]["processed_dir"])
rows = []
with torch.no_grad():
    for i, (g, gt) in enumerate(ds.items):
        bt = {k: v.to(device) for k, v in batch_graphs([g], emb).items()}
        pred = model(bt, build_plan([g]))["boxes"].float().cpu()
        iou, _ = box_iou_giou(pred, gt)
        terms = relation_terms(pred, build_constraints([g]), lc)
        ok = all(bool((v <= 1e-9).all()) for v in terms.values())
        rows.append({"i": i, "g": g, "gt": gt, "pred": pred, "iou": float(iou.mean()), "ok": ok,
                     "maxdepth": max(g.depth_of())})

out = Path("outputs") / a.tag / "figures"
out.mkdir(parents=True, exist_ok=True)
big = [r for r in rows if r["g"].n >= 3]
rnd = random.Random(0)
groups = {
    "best": sorted(big, key=lambda r: -r["iou"])[:a.k],
    "worst": sorted(big, key=lambda r: r["iou"])[:a.k],
    "random": rnd.sample(rows, a.k),
    "multihop": rnd.sample([r for r in rows if r["maxdepth"] >= 2], a.k),
    "violated": [r for r in rows if not r["ok"]][:a.k],
}
for name, rs in groups.items():
    for j, r in enumerate(rs):
        plot_pair(r["gt"], r["pred"], r["g"], r["iou"], r["ok"], out / f"{name}_{j}_idx{r['i']}.png",
                  header=f"[{name} #{r['i']}]")
    print(f"{name}: {len(rs)} figures")
print("saved to", out)
