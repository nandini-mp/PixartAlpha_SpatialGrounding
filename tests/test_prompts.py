"""Stage 3 tests: structure format/parser/validator, prompt generator, round-trip on real data."""
import json
from pathlib import Path

import pytest

from src.utils import load_config
from src.parser.structure import serialize, pretty, parse_structure, BINARY_RELS
from src.data.prompt_gen import build_sample, reverse_parse_prompt, core
from src.graph.relations import check_relation, is_between

CFG = load_config()
PC, RC, SEED = CFG["prompts"], CFG["relations"], CFG["seed"]
PROC = Path(CFG["paths"]["processed_dir"])


def mk(cats, rels, betweens=(), area=0.1):
    objs = [{"id": i, "category": c, "box": [0.5, 0.5, 0.2, 0.2], "area_frac": area,
             "ann_id": i, "category_id": 1} for i, c in enumerate(cats)]
    return {"image_id": 7, "file_name": "x.jpg", "coco_url": "u", "width": 10, "height": 10,
            "objects": objs, "relations": list(rels), "betweens": list(betweens)}


def back_to_orig(struct):
    orig = {o["id"]: o["orig"] for o in struct["objects"]}
    return sorted((r["rel"], tuple(orig[a] for a in r["args"])) for r in struct["relations"])


def expected(rec):
    e = [(r["rel"].lower(), (r["subj"], r["obj"])) for r in rec["relations"]]
    e += [("between_" + t["axis"], (t["mid"], *t["ends"])) for t in rec["betweens"]]
    return sorted(e)


S = {"objects": [{"id": 1, "desc": "small dog"}, {"id": 2, "desc": "person"}, {"id": 3, "desc": "hot dog"}],
     "relations": [{"rel": "left", "args": [1, 2]}, {"rel": "between_x", "args": [3, 1, 2]}]}


def test_structure_roundtrip():
    t = serialize(S)
    assert t == "OBJECTS: 1 = small dog ; 2 = person ; 3 = hot dog RELATIONS: 1 left 2 ; 3 between_x 1 2"
    st, errs, warns = parse_structure(t)
    assert st == S and not errs and not warns
    empty = {"objects": S["objects"], "relations": []}
    assert serialize(empty).endswith("RELATIONS: none")
    assert parse_structure(serialize(empty))[0] == empty


def test_pretty_form_parses():
    st, errs, _ = parse_structure(pretty(S))
    assert st == S and not errs


def test_malformed_output_does_not_crash():
    bad = ["", "garbage", None, 123, "OBJECTS: 1 = dog RELATIONS: 1 left 2",
           "OBJECTS: 1 = dog ; 1 = cat RELATIONS: none",
           "OBJECTS: 1 = dog ; 2 = cat RELATIONS: 1 flies 2",
           "OBJECTS: 1 = dog ; 2 = cat RELATIONS: 1 left",
           "OBJECTS: 1 = dog ; 2 = cat RELATIONS: 1 through 2",
           "OBJECTS: 1 = dog ; 2 = cat RELATIONS: 1 left 1",
           "OBJECTS: 1 = dog ; 2 = cat RELATIONS: 1 between_x 2 2",
           "OBJECTS: dog ; cat RELATIONS: none"]
    for b in bad:
        _, errs, _ = parse_structure(b)
        assert errs, f"expected errors for {b!r}"
    st, errs, _ = parse_structure("OBJECTS: 1 = dog ; 2 = cat RELATIONS: 1 left 2 ; 1 flies 2")
    assert st["relations"] == [{"rel": "left", "args": [1, 2]}] and errs   # partial recovery


def test_contradictions():
    def w(rels):
        return parse_structure(f"OBJECTS: 1 = dog ; 2 = cat RELATIONS: {rels}")[2]
    assert any("contradiction" in x for x in w("1 left 2 ; 1 right 2"))
    assert any("contradiction" in x for x in w("1 left 2 ; 2 left 1"))
    assert any("contradiction" in x for x in w("1 above 2 ; 1 below 2"))
    assert any("contradiction" in x for x in w("1 near 2 ; 2 far 1"))
    assert any("contradiction" in x for x in w("1 inside 2 ; 1 contains 2"))
    assert not any("contradiction" in x for x in w("1 left 2 ; 2 right 1"))   # same fact, not a conflict
    assert not any("contradiction" in x for x in w("1 left 2 ; 1 above 2"))


def test_size_adjectives_and_articles():
    s = build_sample(mk(["dog", "umbrella"], [{"subj": 0, "rel": "LEFT", "obj": 1}], area=0.01), PC, SEED)
    assert {o["desc"] for o in s["struct"]["objects"]} == {"small dog", "small umbrella"}
    s = build_sample(mk(["dog", "umbrella"], [{"subj": 0, "rel": "LEFT", "obj": 1}], area=0.5), PC, SEED)
    assert {o["desc"] for o in s["struct"]["objects"]} == {"large dog", "large umbrella"}
    s = build_sample(mk(["apple", "umbrella"], [{"subj": 0, "rel": "LEFT", "obj": 1}]), PC, SEED)
    p = s["prompt"].lower()
    assert "an apple" in p and "an umbrella" in p


def test_duplicates():
    rec = mk(["person", "person", "dog"], [{"subj": 0, "rel": "LEFT", "obj": 1}])
    for seed in range(200):
        s = build_sample(rec, PC, seed)
        assert reverse_parse_prompt(s["prompt"]) == core(s["struct"])
        assert len(s["struct"]["objects"]) == 3                      # two separate person nodes
        assert s["struct"]["relations"] == [{"rel": "left", "args": [1, 2]}]
        p = s["prompt"].lower()
        assert "the first person" in p and "the second person" in p and "dog" in p


def test_between_and_disconnected():
    rec = mk(["cat", "dog", "cup", "cup", "bottle"],
             [{"subj": 0, "rel": "ABOVE", "obj": 1}],
             [{"mid": 4, "ends": [2, 3], "axis": "x"}])           # 2 components, 1 ternary
    for seed in range(300):
        s = build_sample(rec, PC, seed)
        assert reverse_parse_prompt(s["prompt"]) == core(s["struct"])
        assert back_to_orig(s["struct"]) == expected(rec)
        assert [o["id"] for o in s["struct"]["objects"]] == [1, 2, 3, 4, 5]
        st, errs, warns = parse_structure(s["target"])
        assert st == core(s["struct"]) and not errs


def test_every_relation_phrase_roundtrips():
    for rel in BINARY_RELS:
        rec = mk(["cat", "dog"], [{"subj": 0, "rel": rel.upper(), "obj": 1}])
        for seed in range(60):
            s = build_sample(rec, PC, seed)
            assert reverse_parse_prompt(s["prompt"]) == core(s["struct"])
            assert back_to_orig(s["struct"]) == expected(rec)
    for ax in "xy":
        rec = mk(["cat", "dog", "cup"], [], [{"mid": 2, "ends": [0, 1], "axis": ax}])
        for seed in range(60):
            s = build_sample(rec, PC, seed)
            assert reverse_parse_prompt(s["prompt"]) == core(s["struct"])
            assert back_to_orig(s["struct"]) == expected(rec)


def test_samples_on_real_data():
    pp, rp = PROC / "val_prompts.json", PROC / "val_rel.json"
    if not (pp.exists() and rp.exists()):
        pytest.skip("run scripts/make_prompts.py first")
    samples, recs = json.load(open(pp)), {r["image_id"]: r for r in json.load(open(rp))}
    for s in samples:
        assert reverse_parse_prompt(s["prompt"]) == core(s["struct"])
        st, errs, warns = parse_structure(s["target"])
        assert st == core(s["struct"]) and not errs
        assert not any("contradiction" in w for w in warns)
        n = len(s["boxes"])
        assert [o["id"] for o in s["struct"]["objects"]] == list(range(1, n + 1))
        rec = recs[s["image_id"]]
        assert back_to_orig(s["struct"]) == expected(rec)             # same graph as derived
        for o in s["struct"]["objects"]:
            assert s["boxes"][o["id"] - 1] == rec["objects"][o["orig"]]["box"]
        for r in s["struct"]["relations"]:                            # geometry really satisfies it
            bx = [s["boxes"][a - 1] for a in r["args"]]
            if r["rel"].startswith("between_"):
                assert is_between(bx[0], bx[1], bx[2], r["rel"][-1], RC)
            else:
                assert check_relation(r["rel"].upper(), bx[0], bx[1], RC)
