"""Stage 10: dataset of (LayoutGraph, GT boxes) and a collate that builds all batched tensors."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.graph.batch import batch_graphs
from src.graph.graph import build_graph
from src.graph.plan import build_plan
from src.losses.layout_losses import build_constraints


class LayoutDataset(Dataset):
    def __init__(self, split, processed_dir, limit=None):
        data = json.load(open(Path(processed_dir) / f"{split}_prompts.json"))
        if limit:
            data = data[:limit]
        self.items = []
        for s in data:
            st = {"objects": [{"id": o["id"], "desc": o["desc"]} for o in s["struct"]["objects"]],
                  "relations": s["struct"]["relations"]}
            boxes = torch.tensor(s["boxes"], dtype=torch.float32)
            self.items.append((build_graph(st), boxes))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(embedder):
    def collate(items):
        graphs = [g for g, _ in items]
        return {"graphs": graphs,
                "batch": batch_graphs(graphs, embedder),
                "plan": build_plan(graphs),
                "cons": build_constraints(graphs),
                "gt": torch.cat([b for _, b in items], 0)}
    return collate


def make_loader(ds, embedder, batch_size, shuffle, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(embedder),
                      num_workers=0, generator=g)
