"""Stage 4: fine-tune T5-small on prompt -> structure. Teacher forcing + token cross-entropy.
Validates every epoch (generation-based metrics), keeps best + latest checkpoints, supports --resume.
NEVER touches the test set.

  python scripts/train_parser.py                      # full run
  python scripts/train_parser.py --resume             # continue from checkpoints/parser/latest.pt
  python scripts/train_parser.py --tag smoke --epochs 1 --max_train 640 --max_val 200   # quick test
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (T5ForConditionalGeneration, T5TokenizerFast,
                          get_linear_schedule_with_warmup)

from src.parser.t5_parser import (ParserDataset, evaluate_parser, load_samples, make_collate,
                                  amp_ctx, teacher_forced_loss)
from src.utils import get_device, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_val", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    pc, tc = cfg["parser"], cfg["parser_train"]
    set_seed(cfg["seed"])
    dev = get_device()
    name = "parser" + (f"_{args.tag}" if args.tag else "")
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"]) / name
    out_dir = Path(cfg["paths"]["outputs_dir"]) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)

    tok = T5TokenizerFast.from_pretrained(pc["model_name"])
    train, val = load_samples(cfg, "train"), load_samples(cfg, "val")   # test is never loaded here
    if args.max_train: train = train[:args.max_train]
    if args.max_val: val = val[:args.max_val]
    train_ds = ParserDataset(train, tok, pc["max_source_len"], pc["max_target_len"])
    val_ds = ParserDataset(val, tok, pc["max_source_len"], pc["max_target_len"])
    collate = make_collate(tok.pad_token_id)
    val_loader = DataLoader(val_ds, batch_size=tc["eval_batch_size"], shuffle=False, collate_fn=collate)

    model = T5ForConditionalGeneration.from_pretrained(pc["model_name"]).to(dev)
    epochs = args.epochs or pc["epochs"]
    steps_per_epoch = math.ceil(len(train_ds) / pc["batch_size"])
    total_steps = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=pc["lr"], weight_decay=tc["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(tc["warmup_ratio"] * total_steps), total_steps)

    start, best, best_loss, best_epoch = 1, -1.0, float("inf"), 0
    latest = ckpt_dir / "latest.pt"
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f"--resume given but {latest} does not exist")
        ck = torch.load(latest, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, best_loss, best_epoch = ck["epoch"] + 1, ck["best"], ck["best_loss"], ck["best_epoch"]
        print(f"Resumed from epoch {ck['epoch']} (best {tc['select_metric']}={best:.4f})")

    print(f"device={dev} | train={len(train_ds)} val={len(val_ds)} | epochs={epochs} "
          f"steps/epoch={steps_per_epoch} | bf16={tc['bf16']} | ckpt dir={ckpt_dir}")
    log_path = out_dir / "train_log.jsonl"

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        g = torch.Generator().manual_seed(cfg["seed"] + epoch)      # reproducible per-epoch shuffle
        loader = DataLoader(train_ds, batch_size=pc["batch_size"], shuffle=True, generator=g,
                            collate_fn=collate, drop_last=False)
        model.train()
        run, nb = 0.0, 0
        for b in tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            b = {k: v.to(dev) for k, v in b.items()}
            with amp_ctx(dev, tc["bf16"]):
                loss = model(**b).loss                              # teacher forcing: labels shifted right
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss. Try parser_train.bf16: false or a lower parser.lr.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += loss.item(); nb += 1
        train_loss = run / nb

        vloss = teacher_forced_loss(model, val_loader, dev, tc["bf16"])
        m, _ = evaluate_parser(model, tok, val, dev, pc, tc, tc["num_beams_val"])
        sel = m[tc["select_metric"]]
        is_best = sel > best or (sel == best and vloss < best_loss)
        if is_best:
            best, best_loss, best_epoch = sel, vloss, epoch
            model.save_pretrained(ckpt_dir / "best"); tok.save_pretrained(ckpt_dir / "best")
            json.dump({"epoch": epoch, "val_loss": vloss, "metrics": m}, open(ckpt_dir / "best" / "meta.json", "w"), indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "best_loss": best_loss, "best_epoch": best_epoch}, latest)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": vloss,
               **{k: v for k, v in m.items() if k != "per_relation_recall"}}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch:2d}/{epochs} | train_loss {train_loss:.4f} | val_loss {vloss:.4f} | "
              f"obj {m['object_acc']:.3f} rel {m['relation_acc']:.3f} count {m['count_acc']:.3f} "
              f"exact {m['struct_exact']:.3f} malformed {m['malformed_rate']:.3f} | "
              f"{time.time() - t0:.0f}s{'  *best*' if is_best else ''}")

    print(f"\nDONE. Best {tc['select_metric']}={best:.4f} at epoch {best_epoch}. Saved: {ckpt_dir / 'best'}")


if __name__ == "__main__":
    main()
