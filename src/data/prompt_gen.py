"""Stage 3: natural-language prompts + structured targets from derived COCO relations.

IDs are assigned in order of first mention. Same-description objects are called
"the first X", "the second X", ... (ordinal = rank in ID order).
reverse_parse_prompt() is an independent regex inverse of the templates, used to prove
that every prompt matches its target exactly.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.parser.structure import BINARY_RELS, serialize, pretty

ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
STYLES = ["sentences", "comma", "semicolon"]

PHRASES = {
    "left": ["is to the left of", "is left of", "is on the left of"],
    "right": ["is to the right of", "is right of", "is on the right of"],
    "above": ["is above", "is over", "is on top of"],
    "below": ["is below", "is under", "is beneath"],
    "near": ["is near", "is close to", "is next to"],
    "far": ["is far from", "is far away from"],
    "overlap": ["overlaps", "overlaps with"],
    "inside": ["is inside", "is within"],
    "contains": ["contains", "encloses"],
}
BETWEEN_PHRASES = {"x": ["is horizontally between", "lies horizontally between"],
                   "y": ["is vertically between", "lies vertically between"]}
ISOLATED_LEADS = ["there is also", "the image also contains"]

assert set(PHRASES) == set(BINARY_RELS)
PHRASE2REL = {p: r for r, ps in PHRASES.items() for p in ps}
assert len(PHRASE2REL) == sum(len(v) for v in PHRASES.values())
_ALT = "|".join(re.escape(p) for p in sorted(PHRASE2REL, key=len, reverse=True))
BIN_RE = re.compile(rf"(?P<s>.+?) (?P<p>{_ALT}) (?P<o>.+)")
_BALT = "|".join(re.escape(p) for p in sorted(sum(BETWEEN_PHRASES.values(), []), key=len, reverse=True))
BETWEEN_RE = re.compile(rf"(?P<m>.+?) (?P<p>{_BALT}) (?P<a>.+?) and (?P<b>.+)")
ISO_RE = re.compile("(?:" + "|".join(re.escape(x) for x in ISOLATED_LEADS) + r") (?P<o>.+)")
NP_RE = re.compile(r"(a|an|the) (?:(" + "|".join(ORD) + r") )?(.+)")


def article(desc):
    return "an" if desc[0] in "aeiou" else "a"


def describe(obj, pcfg):
    """Object description = optional size adjective (from box area) + COCO category."""
    a = obj["area_frac"]
    adj = "small" if a < pcfg["size_small_area"] else ("large" if a > pcfg["size_large_area"] else "")
    return f"{adj} {obj['category']}".strip()


def core(struct):
    """Strip a sample struct down to what the parser can recover from text alone."""
    return {"objects": [{"id": o["id"], "desc": o["desc"]} for o in struct["objects"]],
            "relations": [{"rel": r["rel"], "args": list(r["args"])} for r in struct["relations"]]}


def build_sample(rec, pcfg, seed, variant=0):
    """rec: output of derive_record. Returns prompt, target, struct and boxes (ordered by new ID)."""
    rng = np.random.RandomState((seed * 7919 + rec["image_id"] * 31 + variant * 104729 + 17) % (2 ** 32))
    pick = lambda lst: lst[rng.randint(len(lst))]
    objs, n = rec["objects"], len(rec["objects"])
    descs = [describe(o, pcfg) for o in objs]

    # clauses in text order: (kind, [original object indices]); relations first, then isolated objects
    clauses = [(r["rel"].lower(), [r["subj"], r["obj"]]) for r in rec["relations"]]
    clauses += [("between_" + t["axis"], [t["mid"], *t["ends"]]) for t in rec["betweens"]]
    clauses = [clauses[i] for i in rng.permutation(len(clauses))]
    used = {a for _, args in clauses for a in args}
    iso = [int(x) for x in rng.permutation([i for i in range(n) if i not in used])]
    clauses += [("isolated", [i]) for i in iso]

    # IDs = order of first mention (args are rendered in the order listed)
    order = []
    for _, args in clauses:
        for a in args:
            if a not in order:
                order.append(a)
    assert len(order) == n, "every object must be mentioned"
    new_id = {orig: k + 1 for k, orig in enumerate(order)}

    dup = Counter(descs)
    rank, cnt = {}, Counter()
    for orig in order:
        cnt[descs[orig]] += 1
        rank[orig] = cnt[descs[orig]]
    mentioned = set()

    def np_(i):
        d = descs[i]
        if dup[d] > 1:
            s = f"the {ORD[rank[i] - 1]} {d}"
        elif i in mentioned:
            s = f"the {d}"
        else:
            s = f"{article(d)} {d}"
        mentioned.add(i)
        return s

    texts, relations = [], []
    for kind, args in clauses:
        if kind == "isolated":
            lead = pick(ISOLATED_LEADS)
            texts.append(f"{lead} {np_(args[0])}")
        elif kind.startswith("between_"):
            phrase = pick(BETWEEN_PHRASES[kind[-1]])
            m_, a_, b_ = np_(args[0]), np_(args[1]), np_(args[2])
            texts.append(f"{m_} {phrase} {a_} and {b_}")
            relations.append({"rel": kind, "args": [new_id[x] for x in args]})
        else:
            s_ = np_(args[0])
            phrase = pick(PHRASES[kind])
            o_ = np_(args[1])
            texts.append(f"{s_} {phrase} {o_}")
            relations.append({"rel": kind, "args": [new_id[x] for x in args]})

    w = pcfg["style_weights"]
    style = STYLES[rng.choice(len(STYLES), p=[w[s] for s in STYLES])]
    cap = lambda s: s[0].upper() + s[1:]
    if style == "sentences":
        prompt = " ".join(cap(t) + "." for t in texts)
    elif style == "comma":
        prompt = cap(texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + ", and " + texts[-1]) + "."
    else:
        prompt = cap("; ".join(texts)) + "."

    struct = {"objects": [{"id": new_id[o], "desc": descs[o], "category": objs[o]["category"], "orig": o}
                          for o in order],
              "relations": relations}
    return {"image_id": rec["image_id"], "file_name": rec["file_name"], "coco_url": rec["coco_url"],
            "width": rec["width"], "height": rec["height"], "variant": variant, "style": style,
            "prompt": prompt, "target": serialize(struct), "struct": struct,
            "boxes": [objs[o]["box"] for o in order]}


def reverse_parse_prompt(prompt):
    """Independent inverse of the generator (regex only). Raises ValueError on any inconsistency."""
    text = prompt.strip().lower()
    if not text.endswith("."):
        raise ValueError("prompt must end with '.'")
    clauses = [c.strip() for c in re.split(r"\.\s+|, and |,\s*|;\s*", text[:-1]) if c.strip()]
    ids, objects, relations = {}, [], []

    def np_id(s):
        m = NP_RE.fullmatch(s)
        if m is None:
            raise ValueError(f"bad noun phrase {s!r}")
        art, ordw, desc = m.groups()
        ordn = ORD.index(ordw) + 1 if ordw else 0
        key = (desc, ordn)
        if key in ids:
            if art != "the":
                raise ValueError(f"indefinite article on repeated mention {s!r}")
        else:
            if ordn == 0 and art != article(desc):
                raise ValueError(f"first mention must be 'a/an': {s!r}")
            if ordn > 0 and art != "the":
                raise ValueError(f"ordinal phrase must use 'the': {s!r}")
            if ordn > 1 and (desc, ordn - 1) not in ids:
                raise ValueError(f"ordinal order violated: {s!r}")
            ids[key] = len(ids) + 1
            objects.append({"id": ids[key], "desc": desc})
        return ids[key]

    for c in clauses:
        m = ISO_RE.fullmatch(c)
        if m:
            np_id(m["o"])
            continue
        m = BETWEEN_RE.fullmatch(c)
        if m:
            axis = "x" if "horizontally" in m["p"] else "y"
            args = [np_id(m["m"]), np_id(m["a"]), np_id(m["b"])]
            relations.append({"rel": f"between_{axis}", "args": args})
            continue
        m = BIN_RE.fullmatch(c)
        if m:
            s = np_id(m["s"])
            o = np_id(m["o"])
            relations.append({"rel": PHRASE2REL[m["p"]], "args": [s, o]})
            continue
        raise ValueError(f"unparseable clause {c!r}")

    for (desc, ordn) in list(ids):                     # ordinals used consistently
        if ordn > 0 and (desc, 0) in ids:
            raise ValueError(f"mixed plain/ordinal mentions of {desc!r}")
        if ordn == 1 and (desc, 2) not in ids:
            raise ValueError(f"'first {desc}' without a second one")
    return {"objects": objects, "relations": relations}


def run(cfg):
    pcfg, seed = cfg["prompts"], cfg["seed"]
    proc = Path(cfg["paths"]["processed_dir"])
    for split in ("train", "val", "test"):
        recs = json.load(open(proc / f"{split}_rel.json"))
        out = [build_sample(r, pcfg, seed, v) for r in recs
               for v in range(pcfg["variants_per_image"][split])]
        bad = 0
        for s in out:                                   # round-trip proof: prompt text -> struct
            try:
                ok = reverse_parse_prompt(s["prompt"]) == core(s["struct"])
            except ValueError:
                ok = False
            bad += (not ok)
        if bad:
            raise RuntimeError(f"[{split}] {bad} prompts do NOT round-trip to their targets")
        json.dump(out, open(proc / f"{split}_prompts.json", "w"))
        dup = np.mean([len({o['desc'] for o in s['struct']['objects']}) < len(s['struct']['objects'])
                       for s in out])
        adj = np.mean([any(o["desc"].startswith(("small", "large")) for o in s["struct"]["objects"])
                       for s in out])
        words = [len(s["prompt"].split()) for s in out]
        print(f"[{split}] samples={len(out)} round-trip failures=0 | frac_with_duplicates={dup:.3f} "
              f"frac_with_size_adj={adj:.3f} | words/prompt mean={np.mean(words):.1f} max={max(words)} | "
              f"styles={dict(Counter(s['style'] for s in out))}")
    print("\n--- examples (val) ---")
    for s in json.load(open(proc / "val_prompts.json"))[:3]:
        print("PROMPT:", s["prompt"])
        print(pretty(s["struct"]))
        print("TARGET:", s["target"], "\n")
