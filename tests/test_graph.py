"""Stage 5 tests: graph building, inverse edges, components, anchors, BFS multi-hop, BETWEEN, duplicates."""
import json
from pathlib import Path

import pytest

from src.graph.graph import build_graph, describe_graph
from src.parser.structure import parse_structure
from src.utils import load_config


def G(text):
    st, errs, _ = parse_structure(text)
    assert st is not None and not errs, errs
    return build_graph(st)


def rels(g, s, d):
    return [r for a, b, r in g.edges if (a, b) == (s, d)]


def test_single_object():
    g = G("OBJECTS: 1 = dog RELATIONS: none")
    assert g.components == [[0]] and g.anchors == [0]
    assert g.bfs == [[(0, -1, None, 0)]]


def test_two_objects_left_inverse_edges():
    g = G("OBJECTS: 1 = dog ; 2 = cat RELATIONS: 1 left 2")
    assert rels(g, 0, 1) == ["LEFT"] and rels(g, 1, 0) == ["RIGHT"]
    assert g.anchors == [0]                          # equal degree -> smallest index
    # child 1 (cat) relation from parent 0: "cat RIGHT dog"
    assert g.bfs[0] == [(0, -1, None, 0), (1, 0, "RIGHT", 1)]


def test_symmetric_and_inside():
    g = G("OBJECTS: 1 = a ; 2 = b ; 3 = c RELATIONS: 1 near 2 ; 3 inside 1")
    assert rels(g, 0, 1) == ["NEAR"] and rels(g, 1, 0) == ["NEAR"]
    assert rels(g, 2, 0) == ["INSIDE"] and rels(g, 0, 2) == ["CONTAINS"]


def test_multihop_chain_a_above_b_left_c_below_d():
    # A above B, B left C, C below D  (nodes 0..3). Anchor ties -> BFS from a middle node.
    g = G("OBJECTS: 1 = A ; 2 = B ; 3 = C ; 4 = D RELATIONS: 1 above 2 ; 2 left 3 ; 3 below 4")
    assert g.anchors == [1]                          # degree 2, smallest index among {1,2}
    depth = g.depth_of()
    assert depth == [1, 0, 1, 2]
    par = g.parent_of()
    assert par == [1, -1, 1, 2]                      # D (3) hangs off C (2), C off B, A off B
    d = {n: (p, r) for n, p, r, _ in g.bfs[0]}
    assert d[0] == (1, "ABOVE")                      # A ABOVE B
    assert d[2] == (1, "RIGHT")                      # C RIGHT B  (inverse of B left C)
    assert d[3] == (2, "ABOVE")                      # D ABOVE C  (inverse of C below D)


def test_chain_from_end_anchor_is_forced_when_degree_ties():
    g = G("OBJECTS: 1 = A ; 2 = B ; 3 = C RELATIONS: 1 left 2 ; 2 left 3")
    assert g.anchors == [1] and g.depth_of() == [1, 0, 1]


def test_star_anchor_is_highest_degree():
    g = G("OBJECTS: 1 = a ; 2 = b ; 3 = c ; 4 = hub RELATIONS: 1 left 4 ; 2 above 4 ; 3 near 4")
    assert g.anchors == [3] and set(g.depth_of()) == {0, 1}


def test_disconnected_components_each_get_anchor_and_bfs():
    g = G("OBJECTS: 1 = a ; 2 = b ; 3 = c ; 4 = d ; 5 = e RELATIONS: 1 above 2 ; 4 right 5")
    assert g.components == [[0, 1], [2], [3, 4]]
    assert g.anchors == [0, 2, 3]
    assert g.comp_of == [0, 0, 1, 2, 2]
    assert sorted(n for comp in g.bfs for n, *_ in comp) == [0, 1, 2, 3, 4]


def test_duplicate_objects_stay_separate_nodes():
    g = G("OBJECTS: 1 = person ; 2 = person ; 3 = dog RELATIONS: 1 left 2 ; 3 near 1")
    assert g.n == 3 and g.descs == ["person", "person", "dog"]
    assert rels(g, 0, 1) == ["LEFT"]


def test_between_expands_to_binary_edges_and_keeps_ternary():
    g = G("OBJECTS: 1 = a ; 2 = b ; 3 = m RELATIONS: 3 between_x 1 2")
    assert g.betweens == [(2, 0, 1, "x")]
    assert rels(g, 2, 0) == ["BETWEEN_X"] and rels(g, 0, 2) == ["BETWEEN_X"]
    assert rels(g, 2, 1) == ["BETWEEN_X"]
    assert len(g.components) == 1 and g.anchors == [2]        # mid has degree 2


def test_cycle_still_traversed_as_tree():
    g = G("OBJECTS: 1 = a ; 2 = b ; 3 = c RELATIONS: 1 left 2 ; 2 left 3 ; 1 left 3")
    assert len(g.bfs[0]) == 3 and g.warnings == []


def test_duplicate_and_inverse_statements_dedupe():
    g = G("OBJECTS: 1 = a ; 2 = b RELATIONS: 1 left 2 ; 2 right 1")
    assert rels(g, 0, 1) == ["LEFT"] and rels(g, 1, 0) == ["RIGHT"]


def test_contradiction_is_warned_not_fatal():
    g = G("OBJECTS: 1 = a ; 2 = b RELATIONS: 1 left 2 ; 1 right 2")
    assert any("contradictory" in w for w in g.warnings)
    assert len(g.bfs[0]) == 2                                 # still traversable


def test_non_contiguous_ids_map_correctly():
    g = G("OBJECTS: 3 = a ; 7 = b RELATIONS: 3 left 7")
    assert g.ids == [3, 7] and rels(g, 0, 1) == ["LEFT"]


def test_empty_rejected():
    with pytest.raises(ValueError):
        build_graph({"objects": [], "relations": []})


def test_every_node_covered_on_real_data():
    p = Path(load_config()["paths"]["processed_dir"]) / "val_prompts.json"
    if not p.exists():
        pytest.skip("run scripts/make_prompts.py first")
    multi, deep, nb = 0, 0, 0
    for s in json.load(open(p)):
        st = {"objects": [{"id": o["id"], "desc": o["desc"]} for o in s["struct"]["objects"]],
              "relations": s["struct"]["relations"]}
        g = build_graph(st)
        assert sorted(n for comp in g.bfs for n, *_ in comp) == list(range(g.n))
        assert len(g.anchors) == len(g.components)
        assert not g.warnings
        multi += len(g.components) > 1
        deep += max(g.depth_of()) >= 2
        nb += bool(g.betweens)
    print(f"\nval graphs: multi-component={multi} depth>=2={deep} with-between={nb}")
    assert multi > 0 and deep > 0 and nb > 0
