"""Distribution of GT (dx, dy, dw, dh) over BFS parent->child pairs in the TRAIN split."""
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from src.graph.graph import build_graph
from src.models.boxes import encode_delta, MAX_LOG_RATIO
from src.utils import load_config

cfg = load_config()
data = json.load(open(Path(cfg["paths"]["processed_dir"]) / "train_prompts.json"))
by_rel, by_depth, anchors = defaultdict(list), defaultdict(list), []
for s in data:
    st = {"objects": [{"id": o["id"], "desc": o["desc"]} for o in s["struct"]["objects"]],
          "relations": s["struct"]["relations"]}
    g = build_graph(st)
    B = torch.tensor(s["boxes"], dtype=torch.float32)
    assert B.min() >= 0 and B.max() <= 1, "boxes must be normalized (cx,cy,w,h) in [0,1]"
    for comp in g.bfs:
        for node, par, rel, dep in comp:
            if par < 0:
                anchors.append(B[node].tolist())
            else:
                d = encode_delta(B[par], B[node]).tolist()
                by_rel[rel].append(d); by_depth[min(dep, 3)].append(d)

A = np.array(anchors)
print(f"anchors: n={len(A)} | mean box cx,cy,w,h = {np.round(A.mean(0), 3)} | std = {np.round(A.std(0), 3)}")
allD = np.array([d for v in by_rel.values() for d in v])
print(f"non-anchor pairs: {len(allD)} | |dw|>{MAX_LOG_RATIO} or |dh|>{MAX_LOG_RATIO}: "
      f"{np.mean((np.abs(allD[:, 2]) > MAX_LOG_RATIO) | (np.abs(allD[:, 3]) > MAX_LOG_RATIO)) * 100:.2f}%")
print("percentiles [p1, p50, p99] of dx, dy, dw, dh")
for name, grp in [("by relation", by_rel), ("by depth", by_depth)]:
    print(f"--- {name} ---")
    for k in sorted(grp, key=str):
        D = np.array(grp[k])
        print(f"{str(k):10s} n={len(D):6d} " + " | ".join(
            f"{c}:" + str(np.round(np.percentile(D[:, i], [1, 50, 99]), 2)) for i, c in enumerate(["dx", "dy", "dw", "dh"])))
