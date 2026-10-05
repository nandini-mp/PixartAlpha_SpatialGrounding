"""Stage 2: derive spatial relation graphs from COCO boxes (deterministic per image)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.graph.relations import (INVERSE, check_relation, is_between, between_axis,
                                 is_inside, is_overlap, is_near, is_far)


def pair_labels(a, b, rc):
    """For ordered pair (a, b): (primary, [secondaries]); every label r satisfies 'a r b'.
    Returns None if no label applies (practically never)."""
    if is_inside(a, b, rc):
        primary = "INSIDE"
    elif is_inside(b, a, rc):
        primary = "CONTAINS"
    else:
        dx, dy = b[0] - a[0], b[1] - a[1]
        primary = None
        if abs(dx) >= abs(dy):
            if dx > rc["margin"]:    primary = "LEFT"
            elif -dx > rc["margin"]: primary = "RIGHT"
        else:
            if dy > rc["margin"]:    primary = "ABOVE"
            elif -dy > rc["margin"]: primary = "BELOW"
        if primary is None:                       # centers almost identical
            primary = "OVERLAP" if is_overlap(a, b, rc) else ("NEAR" if is_near(a, b, rc) else None)
    if primary is None:
        return None
    sec = []
    for r, fn in (("OVERLAP", is_overlap), ("NEAR", is_near), ("FAR", is_far)):
        if r != primary and fn(a, b, rc):
            sec.append(r)
    return primary, sec


def _components(n, relations, betweens):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = [(r["subj"], r["obj"]) for r in relations]
    for t in betweens:
        edges += [(t["mid"], t["ends"][0]), (t["mid"], t["ends"][1])]
    for u, v in edges:
        parent[find(u)] = find(v)
    return len({find(i) for i in range(n)})


def derive_record(rec, rc, seed):
    """Return a copy of rec with 'relations' and 'betweens' added."""
    rng = np.random.RandomState((seed * 1000003 + rec["image_id"]) % (2 ** 32))
    objs = [dict(o, id=i) for i, o in enumerate(rec["objects"])]
    boxes = [o["box"] for o in objs]
    n = len(objs)

    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            pl = pair_labels(boxes[i], boxes[j], rc)
            if pl:
                pairs[(i, j)] = pl

    def make_edge(i, j):
        primary, sec = pairs[(i, j)]
        rel = primary
        if sec and rng.rand() < rc["secondary_prob"]:
            rel = sec[rng.randint(len(sec))]
        if rng.rand() < 0.5:
            return (i, rel, j)
        return (j, INVERSE[rel], i)               # flipped orientation, inverse relation

    max_rel = rc["max_relations_per_image"]
    chosen = []
    keys = sorted(pairs.keys())
    if keys and rng.rand() < rc["p_connected"]:
        order = [int(x) for x in rng.permutation(n)]
        for k in range(1, n):                     # random spanning tree
            j, i = order[k], order[rng.randint(k)]
            key = (min(i, j), max(i, j))
            if key in pairs:
                chosen.append(key)
        rest = [keys[x] for x in rng.permutation(len(keys)) if keys[x] not in chosen]
        for key in rest:                          # extra edges
            if len(chosen) >= max_rel:
                break
            if rng.rand() < rc["extra_edge_prob"]:
                chosen.append(key)
    elif keys:
        kmax = min(max_rel, len(keys))
        if n >= 3:
            kmax = min(kmax, n - 2)      # <= n-2 edges guarantees >= 2 components
        k = rng.randint(1, kmax + 1)
        chosen = [keys[x] for x in rng.permutation(len(keys))[:k]]

    relations = []
    for (i, j) in sorted(chosen):
        s, r, o = make_edge(i, j)
        assert check_relation(r, boxes[s], boxes[o], rc), (rec["image_id"], s, r, o)
        relations.append({"subj": s, "rel": r, "obj": o})

    betweens = []
    if n >= 3 and rng.rand() < rc["between_prob"]:
        cands = []
        for m in range(n):
            others = [x for x in range(n) if x != m]
            for ai in range(len(others)):
                for bi in range(ai + 1, len(others)):
                    a, b = others[ai], others[bi]
                    ax = between_axis(boxes[a], boxes[b])
                    if is_between(boxes[m], boxes[a], boxes[b], ax, rc):
                        cands.append({"mid": m, "ends": [a, b], "axis": ax})
        if cands:
            betweens.append(cands[rng.randint(len(cands))])

    out = dict(rec)
    out["objects"] = objs
    out["relations"] = relations
    out["betweens"] = betweens
    return out


def run(cfg):
    rc, seed = cfg["relations"], cfg["seed"]
    spl, proc = Path(cfg["paths"]["splits_dir"]), Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for split in ("train", "val", "test"):
        recs = json.load(open(spl / f"{split}.json"))
        out = [derive_record(r, rc, seed) for r in recs]
        json.dump(out, open(proc / f"{split}_rel.json", "w"))
        rel_counts = Counter(x["rel"] for r in out for x in r["relations"])
        ncomp = [_components(len(r["objects"]), r["relations"], r["betweens"]) for r in out]
        stats = {
            "images": len(out),
            "mean_relations": float(np.mean([len(r["relations"]) for r in out])),
            "relation_counts": dict(rel_counts.most_common()),
            "images_with_between": sum(1 for r in out if r["betweens"]),
            "frac_single_component": float(np.mean([c == 1 for c in ncomp])),
            "frac_multi_component": float(np.mean([c > 1 for c in ncomp])),
            "mean_components": float(np.mean(ncomp)),
        }
        all_stats[split] = stats
        print(f"\n[{split}] {json.dumps(stats, indent=2)}")
    json.dump(all_stats, open(proc / "relation_stats.json", "w"), indent=2)
