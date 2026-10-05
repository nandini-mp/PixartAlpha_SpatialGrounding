"""Verifies Stage 1 outputs. Run after scripts/prepare_data.py."""
import json
from pathlib import Path

import pytest
from src.utils import load_config

CFG = load_config()
SPL = Path(CFG["paths"]["splits_dir"])


def _load(name):
    p = SPL / f"{name}.json"
    if not p.exists():
        pytest.skip(f"{p} missing; run scripts/prepare_data.py first")
    return json.load(open(p))


def test_sizes():
    s = CFG["data"]["split"]
    assert len(_load("train")) == s["train"]
    assert len(_load("val")) == s["val"]
    assert len(_load("test")) == s["test"]


def test_no_overlap():
    ids = {k: {r["image_id"] for r in _load(k)} for k in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    assert len(ids["train"] | ids["val"] | ids["test"]) == CFG["data"]["num_samples"]


def test_objects_and_boxes_valid():
    d = CFG["data"]
    for k in ("train", "val", "test"):
        for r in _load(k):
            assert d["min_objects"] <= len(r["objects"]) <= d["max_objects"]
            for o in r["objects"]:
                cx, cy, w, h = o["box"]
                assert w > 0 and h > 0
                assert -1e-4 <= cx - w / 2 and cx + w / 2 <= 1 + 1e-4
                assert -1e-4 <= cy - h / 2 and cy + h / 2 <= 1 + 1e-4
                assert o["area_frac"] >= d["min_box_area_frac"] - 1e-6
