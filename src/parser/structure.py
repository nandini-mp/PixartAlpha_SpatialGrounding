"""Structured representation shared by prompt generator, T5 parser and graph builder.

struct = {"objects":   [{"id": int, "desc": str, ...}],
          "relations": [{"rel": str, "args": [int, ...]}]}
  binary : {"rel": "left", "args": [subj, obj]}       -> "subj left obj"
  ternary: {"rel": "between_x", "args": [mid, a, b]}  -> "mid is between a and b along x"

Serialized form (T5 target) is ONE line because T5's tokenizer cannot represent newlines:
  OBJECTS: 1 = small dog ; 2 = person RELATIONS: 1 left 2 ; 2 between_x 1 3
parse_structure() also accepts the multi-line form produced by pretty().
"""
import re

BINARY_RELS = ["left", "right", "above", "below", "near", "far", "overlap", "inside", "contains"]
TERNARY_RELS = ["between_x", "between_y"]
ARITY = {**{r: 2 for r in BINARY_RELS}, **{r: 3 for r in TERNARY_RELS}}
UNSUPPORTED = {"through"}          # in the vocabulary, but no 2D-box supervision exists

DIRECTIONAL = {"left", "above", "inside"}
FLIP = {"right": "left", "below": "above", "contains": "inside"}   # canonical forms
SYMMETRIC = {"near", "far", "overlap"}


def _rel_line(r):
    return " ".join([str(r["args"][0]), r["rel"], *map(str, r["args"][1:])])


def serialize(struct):
    """Single-line target string (what T5 is trained to generate)."""
    objs = " ; ".join(f"{o['id']} = {o['desc']}" for o in struct["objects"])
    rels = " ; ".join(_rel_line(r) for r in struct["relations"]) or "none"
    return f"OBJECTS: {objs} RELATIONS: {rels}"


def pretty(struct):
    """Multi-line human-readable form."""
    lines = ["OBJECTS:"] + [f"{o['id']} = {o['desc']}" for o in struct["objects"]]
    lines += ["RELATIONS:"] + ([_rel_line(r) for r in struct["relations"]] or ["none"])
    return "\n".join(lines)


def check_contradictions(struct):
    """Return warnings for duplicate or contradictory binary relations."""
    warns, seen, facts = [], set(), []
    for r in struct["relations"]:
        if len(r["args"]) != 2:
            continue
        rel, (a, b) = r["rel"], r["args"]
        if rel in FLIP:
            rel, a, b = FLIP[rel], b, a
        elif rel in SYMMETRIC:
            a, b = sorted((a, b))
        key = (rel, a, b)
        if key in seen:
            warns.append(f"duplicate: {_rel_line(r)}")
            continue
        seen.add(key)
        facts.append(key)
    for rel, a, b in facts:
        if rel in DIRECTIONAL and a < b and (rel, b, a) in seen:
            warns.append(f"contradiction: {rel}({a},{b}) and {rel}({b},{a})")
        if rel == "near" and ("far", a, b) in seen:
            warns.append(f"contradiction: near and far between {a},{b}")
    return warns


def parse_structure(text):
    """Parse + validate a generated structure. NEVER raises.

    Returns (struct_or_None, errors, warnings).
      errors   = format problems (malformed lines, unknown ids/relations, ...). Bad lines are
                 skipped, so struct holds whatever valid part could be recovered.
      warnings = semantic problems (contradictions / duplicate relations).
    struct is None only if nothing usable was found.
    """
    errors, warnings = [], []
    if not isinstance(text, str) or not text.strip():
        return None, ["empty or non-string output"], warnings
    t = text.replace("\n", " ; ")
    m = re.search(r"OBJECTS\s*:(.*?)RELATIONS\s*:(.*)$", t, flags=re.S)
    if m is None:
        return None, ["missing 'OBJECTS:' / 'RELATIONS:' sections"], warnings
    if t[:m.start()].strip(" ;"):
        errors.append("unexpected text before 'OBJECTS:'")

    objects, ids = [], set()
    for seg in (s.strip() for s in m.group(1).split(";")):
        if not seg:
            continue
        mm = re.fullmatch(r"(\d+)\s*=\s*(.+)", seg)
        if mm is None:
            errors.append(f"malformed object line: {seg!r}")
            continue
        oid, desc = int(mm.group(1)), " ".join(mm.group(2).split())
        if oid in ids:
            errors.append(f"duplicate object id {oid}")
            continue
        ids.add(oid)
        objects.append({"id": oid, "desc": desc})
    if not objects:
        return None, errors + ["no valid objects"], warnings

    relations = []
    rel_text = m.group(2).strip(" ;")
    if rel_text and rel_text.lower() != "none":
        for seg in (s.strip() for s in rel_text.split(";")):
            if not seg:
                continue
            tok = seg.split()
            if len(tok) < 3:
                errors.append(f"malformed relation line: {seg!r}")
                continue
            rel = tok[1]
            if rel in UNSUPPORTED:
                errors.append(f"unsupported relation {rel!r}: {seg!r}")
                continue
            if rel not in ARITY:
                errors.append(f"unknown relation name {rel!r}: {seg!r}")
                continue
            arg_tok = [tok[0]] + tok[2:]
            if len(arg_tok) != ARITY[rel] or not all(a.isdigit() for a in arg_tok):
                errors.append(f"malformed relation line: {seg!r}")
                continue
            args = [int(a) for a in arg_tok]
            if any(a not in ids for a in args):
                errors.append(f"relation references unknown object id: {seg!r}")
                continue
            if len(set(args)) != len(args):
                errors.append(f"relation with repeated object (self-relation): {seg!r}")
                continue
            relations.append({"rel": rel, "args": args})

    struct = {"objects": objects, "relations": relations}
    warnings += check_contradictions(struct)
    return struct, errors, warnings
