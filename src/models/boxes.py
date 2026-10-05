"""Box parameterization. Boxes are normalized (cx, cy, w, h) in [0, 1].

Child box from parent box + deltas (Stage 7):
    cx = cx_p + dx * w_p        cy = cy_p + dy * h_p
    w  = w_p * exp(dw)          h  = h_p * exp(dh)
then clamped to a valid box. encode_delta is the exact inverse (when no clamping is active).
"""
import torch

MIN_SIZE = 0.02
MAX_LOG_RATIO = 3.0        # |dw|, |dh| <= 3  ->  size ratio within [1/20, 20]


def decode_box(parent, delta, min_size=MIN_SIZE, max_log_ratio=MAX_LOG_RATIO):
    """parent [..., 4], delta [..., 4] -> child box [..., 4]."""
    cxp, cyp, wp, hp = parent.unbind(-1)
    dx, dy, dw, dh = delta.unbind(-1)
    dw = dw.clamp(-max_log_ratio, max_log_ratio)
    dh = dh.clamp(-max_log_ratio, max_log_ratio)
    cx = (cxp + dx * wp).clamp(0.0, 1.0)
    cy = (cyp + dy * hp).clamp(0.0, 1.0)
    w = (wp * torch.exp(dw)).clamp(min_size, 1.0)
    h = (hp * torch.exp(dh)).clamp(min_size, 1.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def encode_delta(parent, box):
    """Inverse of decode_box: (parent, child box) -> (dx, dy, dw, dh)."""
    cxp, cyp, wp, hp = parent.unbind(-1)
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([(cx - cxp) / wp, (cy - cyp) / hp, torch.log(w / wp), torch.log(h / hp)], dim=-1)
