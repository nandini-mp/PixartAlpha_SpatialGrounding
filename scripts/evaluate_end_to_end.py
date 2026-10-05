"""Stage 13: prompt -> T5 parse -> graph -> layout model, scored against the TRUE GT constraints (run once)."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.data.layout_dataset import LayoutDataset
from src.graph.batch import batch_graphs
from src.graph.graph import build_graph
from src.graph.plan import build_plan
from src.losses.layout_losses import box_iou_giou, build_constraints, relation_terms
from src.models.layout_model import LayoutModel
from src.models.text_encoder import get_embedder
from src.parser.parser_eval import score_prediction
from src.parser.structure import parse_structure
from src.parser.t5_parser import core, generate_texts, load_parser
from src.utils import get_device, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="layout")
ap.add_argument("--parser_ckpt", default="checkpoints/parser/best")
ap.add_argument("--beams", type=int, default=None)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

cfg = load_config(); lc = cfg["layout_loss"]; pc, ptc = cfg["parser"], cfg["parser_train"]
proc = Path(cfg["paths"]["processed_dir"])
out_path = Path("outputs") / a.tag / "e2e_report.json"
if out_path.exists() and not a.force:
    sys.exit(f"{out_path} exists. Evaluate once; use --force only after fixing a bug.")

device = get_device()
samples = json.load(open(proc / "test_prompts.json"))
ds = LayoutDataset("test", proc)
assert len(ds) == len(samples)

# ---- 1. parse every test prompt
ptok_model, tok = load_parser(a.parser_ckpt, device)
beams = a.beams or ptc["num_beams_final"]
texts = generate_texts(ptok_model, tok, [s["prompt"] for s in samples], device,
                       batch_size=ptc["eval_batch_size"], max_source_len=pc["max_source_len"],
                       max_new_tokens=pc["max_target_len"], num_beams=beams, bf16=ptc["bf16"])
del ptok_model
print(f"parsed {len(texts)} prompts with beam={beams}")

# ---- 2. build predicted graphs
recs = []
for s, (gg, gt), t in zip(samples, ds.items, texts):
    score, pred, errs, _ = score_prediction(t, core(s["struct"]))
    rec = {"n": gg.n, "gg": gg, "gt": gt, "exact": bool(score["struct_exact"]), "pg": None, "status": "parse_fail"}
    if pred is not None:
        try:
            rec["pg"] = build_graph({"objects": [{"id": o["id"], "desc": o["desc"]} for o in pred["objects"]],
                                     "relations": pred["relations"]})
        except (ValueError, KeyError):
            rec["pg"] = None
    if rec["pg"] is not None:
        if rec["exact"]:
            rec["status"] = "exact"
        elif rec["pg"].n == gg.n:
            rec["status"] = "parsed_nonexact"
        else:
            rec["status"] = "count_mismatch"
    recs.append(rec)

# ---- 3. layout model
emb = get_embedder(cfg, device)
ck = torch.load(Path("checkpoints") / a.tag / "best.pt", map_location=device, weights_only=False)
tc = ck["cfg"]
model = LayoutModel(emb.dim, tc["hidden"], tc["heads"], tc["layers"], tc["rel_dim"],
                    tc["mlp_hidden"], tc["dropout"]).to(device)
model.load_state_dict(ck["model"]); model.eval()


@torch.no_grad()
def predict(graphs):
    out = []
    for i in range(0, len(graphs), 128):
        chunk = graphs[i:i + 128]
        bt = {k: v.to(device) for k, v in batch_graphs(chunk, emb).items()}
        boxes = model(bt, build_plan(chunk))["boxes"].float().cpu()
        off = 0
        for g in chunk:
            out.append(boxes[off:off + g.n]); off += g.n
    return out


idx_ok = [i for i, r in enumerate(recs) if r["status"] in ("exact", "parsed_nonexact")]
pred_boxes = dict(zip(idx_ok, predict([recs[i]["pg"] for i in idx_ok])))
gt_boxes = predict([r["gg"] for r in recs])           # layout model on GT graphs (Stage 11 setting)


def align(r, pb):
    """Reorder predicted-graph boxes into GT node order (by id if id sets match, else by position)."""
    gt_ids, pids = r["gg"].ids, r["pg"].ids
    if set(gt_ids) == set(pids):
        pos = {pid: k for k, pid in enumerate(pids)}
        return pb[[pos[i] for i in gt_ids]]
    return pb


# ---- 4. score against TRUE GT
def score(r, pred):
    cons = build_constraints([r["gg"]])
    n_rel = sum(len(v) for v in relation_terms(r["gt"], cons, lc).values())
    if pred is None:
        return torch.zeros(r["n"]), 0, n_rel
    iou, _ = box_iou_giou(pred, r["gt"])
    terms = relation_terms(pred, cons, lc)
    return iou, sum(int((v <= 1e-9).sum()) for v in terms.values()), n_rel


rows = []
for i, r in enumerate(recs):
    pred = align(r, pred_boxes[i]) if i in pred_boxes else None
    iou, sat, tot = score(r, pred)
    giou_, gsat, _ = score(r, gt_boxes[i])
    rows.append({"n": r["n"], "status": r["status"], "iou": iou, "sat": sat, "tot": tot,
                 "gt_iou": giou_, "gt_sat": gsat})


def agg(sel, gt_graph=False):
    sel = list(sel)
    if not sel:
        return None
    ik, sk = ("gt_iou", "gt_sat") if gt_graph else ("iou", "sat")
    iou = torch.cat([x[ik] for x in sel]).mean().item()
    tot = sum(x["tot"] for x in sel)
    return {"graphs": len(sel), "iou": iou, "rel_sat": sum(x[sk] for x in sel) / max(1, tot),
            "all_satisfied": sum(x[sk] == x["tot"] for x in sel) / len(sel)}


status_counts = defaultdict(int)
for x in rows:
    status_counts[x["status"]] += 1
exact = [x for x in rows if x["status"] == "exact"]
rep = {"n": len(rows), "beams": beams, "status_counts": dict(status_counts),
       "all": agg(rows), "exact_subset": agg(exact), "exact_subset_gt_graph_run": agg(exact, True),
       "nonexact": agg(x for x in rows if x["status"] != "exact"),
       "gt_graph_all": agg(rows, True),
       "by_n": {n: {"e2e": agg(x for x in rows if x["n"] == n), "gt_graph": agg((x for x in rows if x["n"] == n), True)}
                for n in sorted({x["n"] for x in rows})}}
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(out_path, "w"), indent=1)

f = lambda d: "n/a" if d is None else f"graphs={d['graphs']:4d} IoU {d['iou']:.3f} rel_sat {d['rel_sat']:.3f} all_ok {d['all_satisfied']:.3f}"
print(f"\n=== END-TO-END TEST ({len(rows)} prompts) ===")
print("status counts:", dict(status_counts))
print(f"exact parse rate: {len(exact) / len(rows):.4f}")
print("\nall samples, T5 graph (failures = 0)  :", f(rep["all"]))
print("all samples, GT graph (Stage 11 setup):", f(rep["gt_graph_all"]))
print("exact subset, T5 graph                :", f(rep["exact_subset"]))
print("exact subset, GT graph (must match)   :", f(rep["exact_subset_gt_graph_run"]))
print("non-exact samples, T5 graph           :", f(rep["nonexact"]))
print("\n--- by number of objects: T5 graph | GT graph ---")
for n, v in rep["by_n"].items():
    print(f"n={n} {f(v['e2e'])} || IoU {v['gt_graph']['iou']:.3f} rel_sat {v['gt_graph']['rel_sat']:.3f}")
print("saved to", out_path)
