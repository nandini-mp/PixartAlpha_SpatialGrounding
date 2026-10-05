"""T5-small spatial parser helpers shared by train/eval/inference scripts."""
import contextlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.prompt_gen import core
from src.parser.parser_eval import aggregate, score_prediction


def amp_ctx(device, enabled=True):
    """bf16 autocast on CUDA (T5 is unstable in fp16); no-op on CPU."""
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_samples(cfg, split):
    p = Path(cfg["paths"]["processed_dir"]) / f"{split}_prompts.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run scripts/make_prompts.py first.")
    return json.load(open(p))


class ParserDataset(Dataset):
    """Pre-tokenized (prompt -> target) pairs. Tokenizer appends </s> to both."""
    def __init__(self, samples, tok, max_src, max_tgt):
        self.src = tok([s["prompt"] for s in samples], max_length=max_src, truncation=True)["input_ids"]
        self.tgt = tok([s["target"] for s in samples], max_length=max_tgt, truncation=True)["input_ids"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]


def make_collate(pad_id):
    def collate(batch):
        B = len(batch)
        L, T = max(len(b[0]) for b in batch), max(len(b[1]) for b in batch)
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        lab = torch.full((B, T), -100, dtype=torch.long)       # -100 = ignored by the loss
        for i, (s, t) in enumerate(batch):
            ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
            lab[i, :len(t)] = torch.tensor(t)
        return {"input_ids": ids, "attention_mask": mask, "labels": lab}
    return collate


@torch.no_grad()
def teacher_forced_loss(model, loader, device, bf16=True):
    """Token-level cross-entropy with teacher forcing (per-token average)."""
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with amp_ctx(device, bf16):
            out = model(**b)
        ntok = int((b["labels"] != -100).sum())
        tot += float(out.loss.float()) * ntok
        n += ntok
    return tot / max(n, 1)


@torch.no_grad()
def generate_texts(model, tok, prompts, device, batch_size=128, max_source_len=256,
                   max_new_tokens=256, num_beams=1, bf16=True):
    """Autoregressive generation (no teacher forcing). Returns texts in input order."""
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))   # less padding
    out = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_source_len).to(device)
        with amp_ctx(device, bf16):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                 do_sample=False)
        texts = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, t in zip(idx, texts):
            out[i] = t
    return out


def evaluate_parser(model, tok, samples, device, pcfg, ptcfg, num_beams):
    """Generate for every sample, parse, score against the gold structure.
    Returns (aggregate_metrics, per_sample_details)."""
    texts = generate_texts(model, tok, [s["prompt"] for s in samples], device,
                           batch_size=ptcfg["eval_batch_size"], max_source_len=pcfg["max_source_len"],
                           max_new_tokens=pcfg["max_target_len"], num_beams=num_beams,
                           bf16=ptcfg["bf16"])
    details, scores = [], []
    for s, t in zip(samples, texts):
        score, pred, errs, warns = score_prediction(t, core(s["struct"]))
        scores.append(score)
        details.append({"image_id": s["image_id"], "variant": s["variant"], "style": s["style"],
                        "n_objects": len(s["struct"]["objects"]), "prompt": s["prompt"],
                        "target": s["target"], "pred_text": t, "pred_struct": pred,
                        "errors": errs, "warnings": warns, "score": score})
    return aggregate(scores), details


def load_parser(ckpt_dir, device):
    """Load a saved parser (directory written by train_parser.py) -> (model, tokenizer)."""
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Parser checkpoint not found: {ckpt_dir}")
    tok = T5TokenizerFast.from_pretrained(ckpt_dir)
    model = T5ForConditionalGeneration.from_pretrained(ckpt_dir).to(device).eval()
    return model, tok
