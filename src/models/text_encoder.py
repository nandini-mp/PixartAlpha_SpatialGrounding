"""Frozen CLIP text encoder with a per-description cache (open-vocabulary node semantics)."""
from pathlib import Path

import torch
import torch.nn.functional as F


class DescEmbedder:
    """get(list_of_descriptions) -> float32 CPU tensor [N, dim], L2-normalized.
    Unknown descriptions are encoded on demand and cached; save() persists the cache."""

    def __init__(self, model_name, device, cache_path=None, batch_size=256):
        self.model_name, self.device, self.batch_size = model_name, device, batch_size
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache, self._tok, self._model = {}, None, None
        if self.cache_path and self.cache_path.exists():
            blob = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            if blob.get("model") == model_name:
                self.cache = blob["emb"]

    def _load(self):
        if self._model is None:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
            self._tok = CLIPTokenizer.from_pretrained(self.model_name)
            self._model = CLIPTextModelWithProjection.from_pretrained(self.model_name).to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _encode(self, texts):
        self._load()
        enc = self._tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        return F.normalize(self._model(**enc).text_embeds.float(), dim=-1).cpu()

    @property
    def dim(self):
        if not self.cache:
            self.get(["object"])
        return len(next(iter(self.cache.values())))

    def get(self, descs):
        missing = sorted({d for d in descs if d not in self.cache})
        for s in range(0, len(missing), self.batch_size):
            chunk = missing[s:s + self.batch_size]
            for d, e in zip(chunk, self._encode(chunk)):
                self.cache[d] = e
        return torch.stack([self.cache[d] for d in descs]) if descs else torch.zeros(0, 1)

    def save(self):
        if self.cache_path is None:
            raise ValueError("no cache_path configured")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model_name, "emb": self.cache}, self.cache_path)


def get_embedder(cfg, device):
    """Embedder configured from config.yaml (cache in data/processed/desc_embeddings.pt)."""
    path = Path(cfg["paths"]["processed_dir"]) / "desc_embeddings.pt"
    return DescEmbedder(cfg["model"]["text_encoder"], device, cache_path=path)
