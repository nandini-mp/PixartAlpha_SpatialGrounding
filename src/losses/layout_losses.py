"""Stage 9: box losses (L1, GIoU) and differentiable relation losses on decoded boxes.
Boxes are normalized (cx, cy, w, h). y grows downward (ABOVE means smaller cy)."""
import torch
import torch.nn.functional as F

CANON = ["LEFT", "ABOVE", "INSIDE", "NEAR", "FAR", "OVERLAP"]
SYMMETRIC = ("NEAR", "FAR", "OVERLAP")


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def box_iou_giou(a, b, eps=1e-7):
    """Matched pairs a[i] vs b[i], both (cx,cy,w,h). Returns (iou [N], giou [N])."""
    A, B = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    iw = (torch.min(A[:, 2], B[:, 2]) - torch.max(A[:, 0], B[:, 0])).clamp(min=0)
    ih = (torch.min(A[:, 3], B[:, 3]) - torch.max(A[:, 1], B[:, 1])).clamp(min=0)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    iou = inter / (union + eps)
    ew = torch.max(A[:, 2], B[:, 2]) - torch.min(A[:, 0], B[:, 0])
    eh = torch.max(A[:, 3], B[:, 3]) - torch.min(A[:, 1], B[:, 1])
    enc = ew * eh
    return iou, iou - (enc - union) / (enc + eps)


def build_constraints(graphs):
    """Canonical constraints in batched node indices (same offsets as batch_graphs)."""
    pairs = {r: [] for r in CANON}
    bt, bax, off = [], [], 0
    for g in graphs:
        for s, d, rel in g.edges:
            if rel in ("LEFT", "ABOVE", "INSIDE") or (rel in SYMMETRIC and s < d):
                pairs[rel].append((s + off, d + off))
        for mid, a, b, ax in g.betweens:
            bt.append((mid + off, a + off, b + off)); bax.append(0 if ax == "x" else 1)
        off += g.n
    t = lambda x, k: torch.tensor(x, dtype=torch.long).reshape(-1, k)
    return {"pairs": {r: t(v, 2) for r, v in pairs.items()},
            "between": t(bt, 3), "between_axis": torch.tensor(bax, dtype=torch.long)}


def relation_terms(boxes, cons, cfg):
    """dict: relation name -> per-constraint hinge values [K] (0 = satisfied)."""
    m, dev = cfg["margin"], boxes.device
    xy, out = cxcywh_to_xyxy(boxes), {}
    for rel, p in cons["pairs"].items():
        if len(p) == 0:
            continue
        p = p.to(dev)
        s, d, sx, dx = boxes[p[:, 0]], boxes[p[:, 1]], xy[p[:, 0]], xy[p[:, 1]]
        if rel == "LEFT":
            v = F.relu(s[:, 0] - d[:, 0] + m)
        elif rel == "ABOVE":
            v = F.relu(s[:, 1] - d[:, 1] + m)
        elif rel == "INSIDE":
            sl = cfg.get("inside_slack", 0.0)          # GT containment is tolerance-based
            v = (F.relu(dx[:, 0] - sx[:, 0] - sl) + F.relu(dx[:, 1] - sx[:, 1] - sl)
                 + F.relu(sx[:, 2] - dx[:, 2] - sl) + F.relu(sx[:, 3] - dx[:, 3] - sl))
        elif rel == "OVERLAP":
            ow = torch.min(sx[:, 2], dx[:, 2]) - torch.max(sx[:, 0], dx[:, 0])
            oh = torch.min(sx[:, 3], dx[:, 3]) - torch.max(sx[:, 1], dx[:, 1])
            om = cfg.get("overlap_margin", m)
            v = F.relu(om - ow) + F.relu(om - oh)
        else:
            dist = torch.sqrt(((s[:, :2] - d[:, :2]) ** 2).sum(-1) + 1e-12)
            v = F.relu(dist - cfg["near_dist"]) if rel == "NEAR" else F.relu(cfg["far_dist"] - dist)
        out[rel] = v
    bt = cons["between"]
    if len(bt):
        bt, ax = bt.to(dev), cons["between_axis"].to(dev)
        c = boxes[:, :2]
        mid, a, b = c[bt[:, 0], ax], c[bt[:, 1], ax], c[bt[:, 2], ax]
        v = F.relu(torch.min(a, b) - mid) + F.relu(mid - torch.max(a, b))
        out["BETWEEN_X"], out["BETWEEN_Y"] = v[ax == 0], v[ax == 1]
        out = {k: x for k, x in out.items() if len(x)}
    return out


def relation_loss(boxes, cons, cfg):
    terms = relation_terms(boxes, cons, cfg)
    if not terms:
        return boxes.sum() * 0.0
    return torch.cat(list(terms.values())).mean()


def layout_loss(pred, gt, cons, cfg):
    """pred/gt [N,4] decoded boxes. Returns dict with 'total' and logged parts."""
    l1 = F.l1_loss(pred, gt)
    iou, giou = box_iou_giou(pred, gt)
    lg = (1.0 - giou).mean()
    lr = relation_loss(pred, cons, cfg)
    total = cfg["w_l1"] * l1 + cfg["w_giou"] * lg + cfg["w_rel"] * lr
    return {"total": total, "l1": l1.detach(), "giou_loss": lg.detach(), "rel": lr.detach(),
            "iou": iou.detach().mean()}
