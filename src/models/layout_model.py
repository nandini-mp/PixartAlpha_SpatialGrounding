"""Stage 8: GNN -> anchor box per component -> BFS levels -> child boxes from parent boxes."""
import torch
import torch.nn as nn

from src.models.gnn import EdgeAwareGNN
from src.models.layout_heads import AnchorMLP, NonAnchorMLP


class LayoutModel(nn.Module):
    def __init__(self, text_dim, hidden=256, heads=4, layers=3, rel_dim=64, mlp_hidden=256, dropout=0.0):
        super().__init__()
        self.gnn = EdgeAwareGNN(text_dim, hidden, heads, layers, dropout)
        self.anchor = AnchorMLP(hidden, mlp_hidden, dropout)
        self.nonanchor = NonAnchorMLP(hidden, rel_dim, mlp_hidden, dropout)

    def forward(self, batch, plan, gt_boxes=None, tf_ratio=0.0, generator=None):
        """batch: from batch_graphs; plan: from build_plan; gt_boxes [N,4] (needed if tf_ratio>0).
        Teacher forcing is applied only in training mode. Returns dict(boxes, deltas, feat)."""
        dev = batch["x_text"].device
        plan = {k: v.to(dev) for k, v in plan.items()}
        depth, parent, rel, anchor_of = plan["depth"], plan["parent"], plan["rel"], plan["anchor_of"]
        tf = tf_ratio if self.training else 0.0
        if tf > 0 and gt_boxes is None:
            raise ValueError("gt_boxes required when tf_ratio > 0")

        feat = self.gnn(batch["x_text"], batch["edge_index"], batch["edge_type"])
        N = feat.shape[0]
        boxes, deltas = feat.new_zeros(N, 4), feat.new_zeros(N, 4)

        a_idx = (depth == 0).nonzero(as_tuple=True)[0]                  # all component anchors
        boxes = boxes.index_put((a_idx,), self.anchor(feat[a_idx]).to(boxes.dtype))

        for d in range(1, int(depth.max()) + 1):                         # one batch per BFS level
            idx = (depth == d).nonzero(as_tuple=True)[0]
            par, anc = parent[idx], anchor_of[idx]
            pbox, abox = boxes[par], boxes[anc]
            if tf > 0:
                use = (torch.rand(len(idx), generator=generator) < tf).to(dev)[:, None]
                pbox = torch.where(use, gt_boxes[par], pbox)
                abox = torch.where(use, gt_boxes[anc], abox)
            delta, box = self.nonanchor(rel[idx], abox[:, :2], feat[idx], pbox, feat[par])
            boxes = boxes.index_put((idx,), box.to(boxes.dtype))
            deltas = deltas.index_put((idx,), delta.to(deltas.dtype))
        return {"boxes": boxes, "deltas": deltas, "feat": feat}
