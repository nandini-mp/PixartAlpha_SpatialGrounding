"""Tests for the parser metrics (no model needed) + a perfect-prediction check on real data."""
import json
from pathlib import Path

import pytest

from src.data.prompt_gen import core
from src.parser.parser_eval import aggregate, canon_relations, score_prediction
from src.parser.structure import parse_structure, pretty
from src.utils import load_config

GOLD_TXT = "OBJECTS: 1 = dog ; 2 = cat ; 3 = cup RELATIONS: 1 left 2 ; 3 between_x 1 2"
GOLD = parse_structure(GOLD_TXT)[0]


def sc(text):
    return score_prediction(text, GOLD)[0]


def test_perfect():
    s = sc(GOLD_TXT)
    assert s["struct_exact"] == s["string_exact"] == s["object_exact"] == s["relation_exact"] == 1
    assert s["malformed"] == 0


def test_formatting_does_not_matter():
    s = sc(pretty(GOLD))
    assert s["struct_exact"] == 1 and s["string_exact"] == 0


def test_inverse_symmetric_and_between_order_are_equivalent():
    t = "OBJECTS: 1 = dog ; 2 = cat ; 3 = cup RELATIONS: 2 right 1 ; 3 between_x 2 1"
    assert sc(t)["struct_exact"] == 1
    a = {"objects": [], "relations": [{"rel": "near", "args": [1, 2]}]}
    b = {"objects": [], "relations": [{"rel": "near", "args": [2, 1]}]}
    assert canon_relations(a) == canon_relations(b)


def test_wrong_cases():
    assert sc("OBJECTS: 1 = dog ; 2 = cat ; 3 = cup RELATIONS: 1 right 2 ; 3 between_x 1 2")["relation_exact"] == 0
    assert sc("OBJECTS: 1 = dog ; 2 = cat ; 3 = cup RELATIONS: 2 left 1 ; 3 between_x 1 2")["relation_exact"] == 0
    s = sc("OBJECTS: 1 = dog ; 2 = cat RELATIONS: 1 left 2")
    assert s["count_ok"] == 0 and s["struct_exact"] == 0
    s = sc("OBJECTS: 1 = dog ; 2 = bird ; 3 = cup RELATIONS: 1 left 2 ; 3 between_x 1 2")
    assert s["object_exact"] == 0 and s["relation_exact"] == 1 and s["struct_exact"] == 0


def test_malformed_never_crashes():
    for bad in ["", "garbage", None, "OBJECTS: 1 = dog RELATIONS: 1 flies 1"]:
        s, pred, errs, _ = score_prediction(bad, GOLD)
        assert s["malformed"] == 1 and s["struct_exact"] == 0
    s = sc(GOLD_TXT + " ; 1 flies 2")
    assert s["malformed"] == 1 and s["struct_exact"] == 0 and s["relation_exact"] == 1


def test_aggregate():
    m = aggregate([sc(GOLD_TXT), sc("garbage")])
    assert m["n"] == 2 and m["struct_exact"] == 0.5 and m["malformed_rate"] == 0.5
    assert m["per_relation_recall"] == {"between_x": 0.5, "left": 0.5}


def test_perfect_prediction_on_real_val_data():
    p = Path(load_config()["paths"]["processed_dir"]) / "val_prompts.json"
    if not p.exists():
        pytest.skip("run scripts/make_prompts.py first")
    scores = [score_prediction(s["target"], core(s["struct"]))[0] for s in json.load(open(p))]
    m = aggregate(scores)
    assert m["struct_exact"] == m["string_exact"] == m["object_acc"] == m["relation_acc"] == 1.0
    assert m["malformed_rate"] == 0.0
