"""Parser metrics. Compares PARSED structures (not raw strings) after canonicalization,
so formatting differences never count as errors and inverse/symmetric relations match."""
from collections import Counter

from src.parser.structure import FLIP, SYMMETRIC, parse_structure, serialize


def canon_objects(struct):
    return {(o["id"], o["desc"]) for o in struct["objects"]}


def canon_relations(struct):
    """Canonical relation set: 'b right a' -> 'a left b'; near/far/overlap sorted;
    between_*: (mid, sorted ends)."""
    out = set()
    for r in struct["relations"]:
        rel, a = r["rel"], list(r["args"])
        if len(a) == 2:
            if rel in FLIP:
                rel, a = FLIP[rel], [a[1], a[0]]
            elif rel in SYMMETRIC:
                a = sorted(a)
        else:
            a = [a[0]] + sorted(a[1:])
        out.add((rel, tuple(a)))
    return out


def score_prediction(pred_text, gold_struct):
    """Returns (score_dict, pred_struct_or_None, errors, warnings). Never raises."""
    pred, errs, warns = parse_structure(pred_text)
    malformed = pred is None or bool(errs)
    g_obj, g_rel = canon_objects(gold_struct), canon_relations(gold_struct)
    if pred is None:
        p_obj, p_rel, p_n = set(), set(), 0
    else:
        p_obj, p_rel, p_n = canon_objects(pred), canon_relations(pred), len(pred["objects"])
    obj_exact, rel_exact = p_obj == g_obj, p_rel == g_rel
    rel_gold, rel_tp = Counter(), Counter()
    for rel, args in g_rel:
        rel_gold[rel] += 1
        rel_tp[rel] += (rel, args) in p_rel
    score = {
        "object_exact": int(obj_exact), "relation_exact": int(rel_exact),
        "count_ok": int(p_n == len(gold_struct["objects"])),
        "struct_exact": int(obj_exact and rel_exact and not malformed),
        "string_exact": int(isinstance(pred_text, str) and pred_text.strip() == serialize(gold_struct)),
        "malformed": int(malformed), "contradiction": int(any("contradiction" in w for w in warns)),
        "obj_tp": len(p_obj & g_obj), "obj_pred": len(p_obj), "obj_gold": len(g_obj),
        "rel_tp": len(p_rel & g_rel), "rel_pred": len(p_rel), "rel_gold": len(g_rel),
        "rel_gold_by_type": dict(rel_gold), "rel_tp_by_type": dict(rel_tp),
    }
    return score, pred, errs, warns


def _prf(tp, p, g):
    prec, rec = (tp / p if p else 0.0), (tp / g if g else 0.0)
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def aggregate(scores):
    n = len(scores)
    if n == 0:
        return {"n": 0}
    mean = lambda k: sum(s[k] for s in scores) / n
    tot = lambda k: sum(s[k] for s in scores)
    op, orr, of = _prf(tot("obj_tp"), tot("obj_pred"), tot("obj_gold"))
    rp, rr, rf = _prf(tot("rel_tp"), tot("rel_pred"), tot("rel_gold"))
    gold, tp = Counter(), Counter()
    for s in scores:
        gold.update(s["rel_gold_by_type"]); tp.update(s["rel_tp_by_type"])
    return {
        "n": n, "object_acc": mean("object_exact"), "relation_acc": mean("relation_exact"),
        "count_acc": mean("count_ok"), "struct_exact": mean("struct_exact"),
        "string_exact": mean("string_exact"), "malformed_rate": mean("malformed"),
        "contradiction_rate": mean("contradiction"),
        "object_precision": op, "object_recall": orr, "object_f1": of,
        "relation_precision": rp, "relation_recall": rr, "relation_f1": rf,
        "per_relation_recall": {r: tp[r] / gold[r] for r in sorted(gold)},
    }
