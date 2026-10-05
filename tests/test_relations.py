"""Unit tests for relation predicates + sanity checks on derived data."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.utils import load_config
from src.graph.relations import (INVERSE, PREDICATES, check_relation, is_between,
                                 between_axis, iou, RELATIONS)
from src.data.derive_relations import pair_labels, derive_record

CFG = load_config()
RC = CFG["relations"]


def B(cx, cy, w=0.1, h=0.1):
    return [cx, cy, w, h]


def test_left_right():
    a, b = B(0.2, 0.5), B(0.7, 0.5)
    assert check_relation("LEFT", a, b, RC) and not check_relation("LEFT", b, a, RC)
    assert check_relation("RIGHT", b, a, RC) and not check_relation("RIGHT", a, b, RC)


def test_above_below():
    a, b = B(0.5, 0.2), B(0.5, 0.7)          # smaller cy = higher in the image
    assert check_relation("ABOVE", a, b, RC) and not check_relation("ABOVE", b, a, RC)
    assert check_relation("BELOW", b, a, RC) and not check_relation("BELOW", a, b, RC)


def test_margin_tolerance():
    assert not check_relation("LEFT", B(0.50, 0.5), B(0.51, 0.5), RC)   # within margin
    assert check_relation("LEFT", B(0.50, 0.5), B(0.60, 0.5), RC)


def test_overlap():
    assert check_relation("OVERLAP", B(0.4, 0.5, 0.3, 0.3), B(0.5, 0.5, 0.3, 0.3), RC)
    assert not check_relation("OVERLAP", B(0.1, 0.1), B(0.9, 0.9), RC)
    assert abs(iou(B(0.4, 0.5, 0.3, 0.3), B(0.5, 0.5, 0.3, 0.3)) - 0.5) < 1e-6


def test_inside_contains():
    small, big = B(0.5, 0.5, 0.1, 0.1), B(0.5, 0.5, 0.6, 0.6)
    assert check_relation("INSIDE", small, big, RC) and check_relation("CONTAINS", big, small, RC)
    assert not check_relation("INSIDE", big, small, RC)
    assert not check_relation("INSIDE", B(0.9, 0.9), big, RC)


def test_near_far():
    assert check_relation("NEAR", B(0.5, 0.5), B(0.6, 0.5), RC)
    assert not check_relation("NEAR", B(0.1, 0.1), B(0.9, 0.9), RC)
    assert check_relation("FAR", B(0.1, 0.1), B(0.9, 0.9), RC)
    assert not check_relation("FAR", B(0.5, 0.5), B(0.6, 0.5), RC)


def test_between():
    a, b = B(0.1, 0.5), B(0.9, 0.5)
    assert between_axis(a, b) == "x"
    assert is_between(B(0.5, 0.5), a, b, "x", RC)
    assert not is_between(B(0.95, 0.5), a, b, "x", RC)        # outside the segment
    assert not is_between(B(0.5, 0.95), a, b, "x", RC)        # far off the line
    assert between_axis(B(0.5, 0.1), B(0.5, 0.9)) == "y"
    assert is_between(B(0.5, 0.5), B(0.5, 0.1), B(0.5, 0.9), "y", RC)


def test_inverse_consistency():
    rng = np.random.RandomState(0)
    for _ in range(300):
        a = [*rng.uniform(0.2, 0.8, 2), *rng.uniform(0.05, 0.4, 2)]
        b = [*rng.uniform(0.2, 0.8, 2), *rng.uniform(0.05, 0.4, 2)]
        for rel, fn in PREDICATES.items():
            if fn(a, b, RC):
                assert PREDICATES[INVERSE[rel]](b, a, RC), (rel, a, b)
    for r in RELATIONS:
        assert INVERSE[INVERSE[r]] == r


def test_through_unsupported():
    with pytest.raises(NotImplementedError):
        check_relation("THROUGH", B(0.3, 0.3), B(0.6, 0.6), RC)


def test_pair_labels_directionality():
    a, b = B(0.2, 0.5), B(0.7, 0.5)
    assert pair_labels(a, b, RC)[0] == "LEFT"
    assert pair_labels(b, a, RC)[0] == "RIGHT"
    assert pair_labels(B(0.5, 0.2), B(0.5, 0.7), RC)[0] == "ABOVE"
    assert pair_labels(B(0.5, 0.5, 0.1, 0.1), B(0.5, 0.5, 0.6, 0.6), RC)[0] == "INSIDE"


def _load_rel(split):
    p = Path(CFG["paths"]["processed_dir"]) / f"{split}_rel.json"
    if not p.exists():
        pytest.skip("run scripts/derive_relations.py first")
    return json.load(open(p))


def test_derived_relations_hold_on_data():
    for split in ("val", "test"):
        for r in _load_rel(split):
            boxes = [o["box"] for o in r["objects"]]
            seen = set()
            for x in r["relations"]:
                s, o = x["subj"], x["obj"]
                assert s != o and 0 <= s < len(boxes) and 0 <= o < len(boxes)
                assert frozenset((s, o)) not in seen, "duplicate pair"
                seen.add(frozenset((s, o)))
                assert check_relation(x["rel"], boxes[s], boxes[o], RC)
            for t in r["betweens"]:
                a, b = t["ends"]
                assert is_between(boxes[t["mid"]], boxes[a], boxes[b], t["axis"], RC)


def test_derivation_deterministic():
    p = Path(CFG["paths"]["splits_dir"]) / "val.json"
    if not p.exists():
        pytest.skip("run scripts/prepare_data.py first")
    recs = json.load(open(p))[:200]
    for r in recs:
        assert derive_record(r, RC, CFG["seed"]) == derive_record(r, RC, CFG["seed"])
