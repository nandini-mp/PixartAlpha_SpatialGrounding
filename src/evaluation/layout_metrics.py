"""Stage 10: layout evaluation (IoU, GIoU, L1, relation satisfaction) for any box predictor."""
from collections import defaultdict

import torch

from src.losses.layout_losses import box_iou_giou, layout_loss, relation_terms


def model_predict(model):
    def f(b, device):
        bt = {k: v.to(device) for k, v in b["batch"].items()}
        return model(bt, b["plan"])["boxes"]
    return f


def mean_box_predict(mean_box):
    def f(b, device):
        return mean_box.to(device).expand(len(b["gt"]), 4).clone()
    return f


@torch.no_grad()
def evaluate(predict, loader, lc, device):
    """predict(batch_dict, device) -> boxes [N,4]. Caller sets model.eval().
    rel_sat = fraction of relation constraints with zero hinge (same definition as the loss)."""
    ious, gious, l1s, losses = [], [], [], []
    sat, tot = defaultdict(int), defaultdict(int)
    for b in loader:
        gt = b["gt"].to(device)
        pred = predict(b, device).float()
        iou, giou = box_iou_giou(pred, gt)
        ious.append(iou.cpu()); gious.append(giou.cpu()); l1s.append((pred - gt).abs().mean(-1).cpu())
        losses.append(float(layout_loss(pred, gt, b["cons"], lc)["total"]))
        for r, v in relation_terms(pred, b["cons"], lc).items():
            sat[r] += int((v <= 1e-9).sum()); tot[r] += len(v)
    n_all = sum(tot.values())
    return {"iou": torch.cat(ious).mean().item(), "giou": torch.cat(gious).mean().item(),
            "l1": torch.cat(l1s).mean().item(), "loss": sum(losses) / max(1, len(losses)),
            "rel_sat": sum(sat.values()) / max(1, n_all),
            "per_rel": {r: sat[r] / tot[r] for r in sorted(tot)}, "n_rel": dict(sorted(tot.items()))}
