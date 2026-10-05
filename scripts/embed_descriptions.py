"""Embed every unique object description with frozen CLIP and cache it."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.models.text_encoder import get_embedder
from src.utils import get_device, load_config

cfg = load_config()
proc = Path(cfg["paths"]["processed_dir"])
descs = sorted({o["desc"] for sp in ("train", "val", "test")
                for s in json.load(open(proc / f"{sp}_prompts.json")) for o in s["struct"]["objects"]})
emb = get_embedder(cfg, get_device())
E = emb.get(descs)
emb.save()
print(f"unique descriptions: {len(descs)} | embedding matrix: {tuple(E.shape)} | "
      f"norm min/max: {E.norm(dim=-1).min():.4f}/{E.norm(dim=-1).max():.4f}")
cos = lambda a, b: float(emb.get([a])[0] @ emb.get([b])[0])
for a, b in [("small dog", "dog"), ("small dog", "small cat"), ("small dog", "large dog"),
             ("small dog", "small car"), ("large person", "person"), ("small dog", "large car")]:
    print(f"cos({a!r}, {b!r}) = {cos(a, b):.3f}")
print("saved to", emb.cache_path)
