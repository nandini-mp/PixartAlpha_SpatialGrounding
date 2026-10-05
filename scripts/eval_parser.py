"""Evaluate a trained parser on val or test; saves predictions for Stage 13.

  python scripts/eval_parser.py --split val
  python scripts/eval_parser.py --split test --final     # test needs --final; run it ONCE
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser.parser_eval import aggregate
from src.parser.t5_parser import evaluate_parser, load_parser, load_samples
from src.utils import get_device, load_config, set_seed


def breakdown(details, key_fn):
    groups = defaultdict(list)
    for d in details:
        groups[key_fn(d)].append(d["score"])
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--tag", default="")
    ap.add_argument("--beams", type=int, default=None)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--final", action="store_true", help="required for the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("Refusing to touch the test set without --final (it must be evaluated once, "
                 "after all model selection is finished).")

    cfg = load_config()
    pc, tc = cfg["parser"], cfg["parser_train"]
    set_seed(cfg["seed"]); dev = get_device()
    name = "parser" + (f"_{args.tag}" if args.tag else "")
    model, tok = load_parser(Path(cfg["paths"]["checkpoints_dir"]) / name / "best", dev)
    samples = load_samples(cfg, args.split)
    if args.max: samples = samples[:args.max]
    beams = args.beams or tc["num_beams_final"]
    m, details = evaluate_parser(model, tok, samples, dev, pc, tc, beams)

    print(f"\n=== T5 parser | split={args.split} | n={m['n']} | beams={beams} ===")
    for k in ["object_acc", "relation_acc", "count_acc", "struct_exact", "string_exact",
              "malformed_rate", "contradiction_rate", "object_f1", "relation_precision",
              "relation_recall", "relation_f1"]:
        print(f"{k:20s}: {m[k]:.4f}")
    print("per-relation recall (right/below/contains are folded into left/above/inside):")
    for r, v in m["per_relation_recall"].items():
        print(f"  {r:10s}: {v:.4f}")
    by_n = breakdown(details, lambda d: d["n_objects"])
    by_style = breakdown(details, lambda d: d["style"])
    print("struct_exact by #objects:", {k: (round(v["struct_exact"], 3), v["n"]) for k, v in by_n.items()})
    print("struct_exact by style   :", {k: (round(v["struct_exact"], 3), v["n"]) for k, v in by_style.items()})

    fails = [d for d in details if not d["score"]["struct_exact"]]
    print(f"\n{len(fails)} failures; first 5:")
    for d in fails[:5]:
        print("PROMPT:", d["prompt"]); print("GOLD  :", d["target"]); print("PRED  :", d["pred_text"])
        print("ERRORS:", d["errors"], "\n")

    out_dir = Path(cfg["paths"]["outputs_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(details, open(out_dir / f"{args.split}_predictions.json", "w"))
    json.dump({"metrics": m, "by_n_objects": by_n, "by_style": by_style, "beams": beams},
              open(out_dir / f"{args.split}_parser_metrics.json", "w"), indent=2)
    print(f"Saved predictions + metrics to {out_dir}")


if __name__ == "__main__":
    main()
