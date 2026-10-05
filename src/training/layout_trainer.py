"""Stage 10: layout training loop (teacher-forcing schedule, validation each epoch, checkpoints, resume)."""
import contextlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from src.data.layout_dataset import LayoutDataset, make_loader
from src.evaluation.layout_metrics import evaluate, mean_box_predict, model_predict
from src.losses.layout_losses import layout_loss
from src.models.layout_model import LayoutModel
from src.models.text_encoder import get_embedder
from src.utils import get_device


def tf_ratio(epoch, tc):
    """Linear decay tf_start -> tf_end over tf_decay_epochs (epoch is 0-based)."""
    f = min(1.0, epoch / max(1, tc["tf_decay_epochs"]))
    return tc["tf_start"] + (tc["tf_end"] - tc["tf_start"]) * f


def lr_factor(step, warmup, total, floor=0.01):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


def _fmt_rel(d):
    return " ".join(f"{k[:6]}={v:.2f}" for k, v in d.items())


def train(cfg, limit=None, epochs=None, resume=False, tag="layout"):
    tc, lc = cfg["layout_train"], cfg["layout_loss"]
    E = epochs or tc["epochs"]
    device = get_device()
    dev_type = torch.device(device).type
    torch.manual_seed(tc["seed"])
    proc = cfg["paths"]["processed_dir"]
    out_dir, ck_dir = Path("outputs") / tag, Path("checkpoints") / tag
    out_dir.mkdir(parents=True, exist_ok=True); ck_dir.mkdir(parents=True, exist_ok=True)

    emb = get_embedder(cfg, device)
    train_ds, val_ds = LayoutDataset("train", proc, limit), LayoutDataset("val", proc, limit)
    train_loader = make_loader(train_ds, emb, tc["batch_size"], True, tc["seed"])
    val_loader = make_loader(val_ds, emb, tc["batch_size"], False)
    mean_box = torch.cat([b for _, b in train_ds.items]).mean(0)
    base = evaluate(mean_box_predict(mean_box), val_loader, lc, device)
    print(f"train={len(train_ds)} val={len(val_ds)} | mean-box baseline on val: IoU {base['iou']:.3f} "
          f"L1 {base['l1']:.3f} rel_sat {base['rel_sat']:.3f}")

    model = LayoutModel(emb.dim, tc["hidden"], tc["heads"], tc["layers"], tc["rel_dim"],
                        tc["mlp_hidden"], tc["dropout"]).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = E * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, tc["warmup_steps"], total_steps))
    use_amp = bool(tc["amp"]) and dev_type == "cuda" and torch.cuda.is_bf16_supported()
    gen = torch.Generator().manual_seed(tc["seed"])

    start, best, bad = 0, float("inf"), 0
    if resume and (ck_dir / "last.pt").exists():
        ck = torch.load(ck_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"resumed from epoch {start}")

    log = open(out_dir / "log.jsonl", "a")
    for epoch in range(start, E):
        model.train()
        tf = tf_ratio(epoch, tc)
        run, nb, skipped = defaultdict(float), 0, 0
        for b in train_loader:
            bt = {k: v.to(device) for k, v in b["batch"].items()}
            gt = b["gt"].to(device)
            ctx = torch.autocast(dev_type, dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with ctx:
                out = model(bt, b["plan"], gt, tf, gen)
            parts = layout_loss(out["boxes"].float(), gt, b["cons"], lc)
            if not torch.isfinite(parts["total"]):
                skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step(); sched.step()
            for k in ("total", "iou", "rel"):
                run[k] += float(parts[k])
            nb += 1
        model.eval()
        val = evaluate(model_predict(model), val_loader, lc, device)
        improved = val["loss"] < best
        if improved:
            best, bad = val["loss"], 0
            torch.save({"model": model.state_dict(), "cfg": tc, "epoch": epoch, "val": val}, ck_dir / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad}, ck_dir / "last.pt")
        rec = {"epoch": epoch, "tf": tf, "lr": sched.get_last_lr()[0], "skipped": skipped,
               "train_loss": run["total"] / max(1, nb), "train_iou": run["iou"] / max(1, nb), "val": val}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"ep {epoch:02d} tf={tf:.2f} | train loss {rec['train_loss']:.3f} iou {rec['train_iou']:.3f} | "
              f"val loss {val['loss']:.3f} IoU {val['iou']:.3f} L1 {val['l1']:.3f} "
              f"rel_sat {val['rel_sat']:.3f}{' *' if improved else ''}"
              + (f" | skipped {skipped}" if skipped else ""))
        if bad >= tc["patience"]:
            print(f"early stop (no val improvement for {bad} epochs)")
            break

    ck = torch.load(ck_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    val = evaluate(model_predict(model), val_loader, lc, device)
    print(f"\n=== best checkpoint (epoch {ck['epoch']}) vs mean-box baseline | val ===")
    print(f"IoU {val['iou']:.3f} (base {base['iou']:.3f}) | GIoU {val['giou']:.3f} (base {base['giou']:.3f}) | "
          f"L1 {val['l1']:.3f} (base {base['l1']:.3f}) | rel_sat {val['rel_sat']:.3f} (base {base['rel_sat']:.3f})")
    for r in val["per_rel"]:
        print(f"  {r:10s} n={val['n_rel'][r]:5d} model sat={val['per_rel'][r]:.3f} baseline sat={base['per_rel'][r]:.3f}")
    json.dump({"model": val, "baseline": base, "epoch": ck["epoch"]}, open(out_dir / "val_report.json", "w"), indent=1)
    return val
