"""Stage 7: anchor head (absolute box) and non-anchor head (deltas relative to the parent box).

NonAnchorMLP inputs (per node):
    relation embedding   relation of the node as seen FROM its parent ("child REL parent")
    anchor cx, cy        global context: where the component's anchor was placed
                         (for depth >= 2 this is supplementary; the parent box does the geometry)
    node feature         GNN feature of the node
    parent box           (cx, cy, w, h) of the parent (predicted at inference; GT optional in training)
    parent feature       GNN feature of the parent
Output: (dx, dy, dw, dh) and the decoded child box.
"""
import torch
import torch.nn as nn

from src.graph.relations import RELATIONS
from src.models.boxes import MIN_SIZE, MAX_LOG_RATIO, decode_box


def _mlp(i, h, o, dropout):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(h, o))


class AnchorMLP(nn.Module):
    """node feature -> absolute box (cx, cy, w, h)."""

    def __init__(self, hidden, mlp_hidden=256, dropout=0.0, min_size=MIN_SIZE, init_size=0.3):
        super().__init__()
        self.net = _mlp(hidden, mlp_hidden, 4, dropout)
        self.min_size = min_size
        last = self.net[-1]
        nn.init.normal_(last.weight, std=1e-3)
        # start as a centered box of about init_size x init_size
        s = (init_size - min_size) / (1 - min_size)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([0.0, 0.0, torch.logit(torch.tensor(s)).item(),
                                          torch.logit(torch.tensor(s)).item()]))

    def forward(self, feat):
        o = torch.sigmoid(self.net(feat))
        wh = self.min_size + (1.0 - self.min_size) * o[..., 2:]
        return torch.cat([o[..., :2], wh], dim=-1)


class NonAnchorMLP(nn.Module):
    def __init__(self, hidden, rel_dim=64, mlp_hidden=256, dropout=0.0,
                 num_relations=len(RELATIONS), min_size=MIN_SIZE, max_log_ratio=MAX_LOG_RATIO):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, rel_dim)
        self.in_dim = rel_dim + 2 + hidden + 4 + hidden
        self.net = _mlp(self.in_dim, mlp_hidden, 4, dropout)
        self.min_size, self.max_log_ratio = min_size, max_log_ratio
        last = self.net[-1]                                   # deltas start near 0 -> child ~ parent box
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)

    def forward(self, rel_ids, anchor_c, node_feat, parent_box, parent_feat):
        """rel_ids [B] long; anchor_c [B,2]; node_feat [B,H]; parent_box [B,4]; parent_feat [B,H]
        -> (delta [B,4], box [B,4])"""
        x = torch.cat([self.rel_emb(rel_ids), anchor_c, node_feat, parent_box, parent_feat], dim=-1)
        delta = self.net(x)
        return delta, decode_box(parent_box, delta, self.min_size, self.max_log_ratio)
