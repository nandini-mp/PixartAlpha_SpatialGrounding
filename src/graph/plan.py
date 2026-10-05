"""Flatten BFS structure of a list of LayoutGraphs into per-node tensors (batched node order,
same offsets as batch_graphs)."""
import torch

from src.graph.relations import REL2ID


def build_plan(graphs):
    depth, parent, rel, anchor_of = [], [], [], []
    off = 0
    for g in graphs:
        d, p, r, a = [0] * g.n, [-1] * g.n, [-1] * g.n, [0] * g.n
        for order, anc in zip(g.bfs, g.anchors):
            for node, par, rl, dep in order:
                d[node] = dep
                p[node] = par + off if par >= 0 else -1
                r[node] = REL2ID[rl] if rl is not None else -1
                a[node] = anc + off
        depth += d; parent += p; rel += r; anchor_of += a
        off += g.n
    t = lambda x: torch.tensor(x, dtype=torch.long)
    return {"depth": t(depth), "parent": t(parent), "rel": t(rel), "anchor_of": t(anchor_of)}
