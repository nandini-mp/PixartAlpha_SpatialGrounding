"""Relation vocabulary + deterministic geometric predicates.

Boxes are (cx, cy, w, h), normalized to [0,1]. Image y grows DOWNWARD.
Predicate p(a, b) answers: "does the statement 'a REL b' hold?"
"""
import math

RELATIONS = ["LEFT", "RIGHT", "ABOVE", "BELOW", "NEAR", "FAR", "OVERLAP",
             "INSIDE", "CONTAINS", "BETWEEN_X", "BETWEEN_Y", "THROUGH"]
REL2ID = {r: i for i, r in enumerate(RELATIONS)}

# Inverse seen from the other endpoint. NEAR/FAR/OVERLAP are symmetric.
# BETWEEN_X/Y (binary expansion of the ternary relation) and THROUGH are only
# mapped to themselves so graph traversal can walk the edge both ways.
INVERSE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "ABOVE": "BELOW", "BELOW": "ABOVE",
           "INSIDE": "CONTAINS", "CONTAINS": "INSIDE", "NEAR": "NEAR", "FAR": "FAR",
           "OVERLAP": "OVERLAP", "BETWEEN_X": "BETWEEN_X", "BETWEEN_Y": "BETWEEN_Y",
           "THROUGH": "THROUGH"}
SYMMETRIC = {"NEAR", "FAR", "OVERLAP"}
# THROUGH cannot be derived from static 2D boxes -> never supervised.
UNSUPERVISED = {"THROUGH"}


# ---------- box geometry ----------
def to_xyxy(b):
    cx, cy, w, h = b
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def area(b):
    return b[2] * b[3]


def inter_area(a, b):
    ax0, ay0, ax1, ay1 = to_xyxy(a)
    bx0, by0, bx1, by1 = to_xyxy(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih


def iou(a, b):
    i = inter_area(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def center_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------- binary predicates: p(a, b, rc) ----------
def is_left(a, b, rc):      return (b[0] - a[0]) > rc["margin"]
def is_right(a, b, rc):     return (a[0] - b[0]) > rc["margin"]
def is_above(a, b, rc):     return (b[1] - a[1]) > rc["margin"]
def is_below(a, b, rc):     return (a[1] - b[1]) > rc["margin"]
def is_overlap(a, b, rc):   return iou(a, b) > rc["overlap_iou"]
def is_near(a, b, rc):      return center_dist(a, b) < rc["near_dist"]
def is_far(a, b, rc):       return center_dist(a, b) > rc["far_dist"]


def is_inside(a, b, rc):
    """a INSIDE b: a is strictly smaller and >= inside_frac of a's area lies in b."""
    aa = area(a)
    return aa < area(b) and aa > 0 and inter_area(a, b) / aa >= rc["inside_frac"]


def is_contains(a, b, rc):  return is_inside(b, a, rc)


PREDICATES = {"LEFT": is_left, "RIGHT": is_right, "ABOVE": is_above, "BELOW": is_below,
              "OVERLAP": is_overlap, "NEAR": is_near, "FAR": is_far,
              "INSIDE": is_inside, "CONTAINS": is_contains}


def check_relation(rel, a, b, rc):
    """Does 'a REL b' hold for boxes a, b? Raises for unsupported relations."""
    if rel not in RELATIONS:
        raise ValueError(f"Unknown relation {rel!r}")
    if rel in UNSUPERVISED:
        raise NotImplementedError(f"{rel} cannot be derived/checked from 2D boxes")
    if rel in ("BETWEEN_X", "BETWEEN_Y"):
        raise ValueError("BETWEEN is ternary: use is_between(m, a, b, axis, rc)")
    return PREDICATES[rel](a, b, rc)


# ---------- ternary BETWEEN ----------
def between_axis(a, b):
    """Deterministic axis for 'between a and b': larger center separation (tie -> x)."""
    return "x" if abs(a[0] - b[0]) >= abs(a[1] - b[1]) else "y"


def is_between(m, a, b, axis, rc):
    """m lies strictly between a and b along `axis` (by margin) and near their
    midpoint on the other axis (within between_perp_tol)."""
    k = 0 if axis == "x" else 1
    p = 1 - k
    lo, hi = sorted((a[k], b[k]))
    if not (lo + rc["margin"] < m[k] < hi - rc["margin"]):
        return False
    return abs(m[p] - (a[p] + b[p]) / 2) <= rc.get("between_perp_tol", 0.25)
