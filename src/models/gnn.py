"""Edge-aware GATv2 in plain PyTorch (no torch-geometric).

Edge (s -> d, REL) means "s REL d". Node d aggregates over its incoming edges:
    msg_sd   = W_src x_s + W_edge e_REL
    logit_sd = a^T LeakyReLU(W_dst x_d + msg_sd)          (per head)
    alpha    = softmax over the incoming edges of d       (scatter-softmax)
    h_d      = sum_s alpha_sd * msg_sd
so the relation affects BOTH the attention weights and the message content.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.relations import RELATIONS


def scatter_softmax(logits, index, n):
    """Softmax of logits [E, H] over groups given by index [E] (n groups). Computed in fp32."""
    logits = logits.float()
    H = logits.shape[1]
    idx = index.unsqueeze(-1).expand(-1, H)
    mx = torch.full((n, H), float("-inf"), device=logits.device).scatter_reduce(0, idx, logits, "amax", include_self=True)
    ex = torch.exp(logits - mx[index])
    den = torch.zeros(n, H, device=logits.device).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)


class EdgeGATv2Layer(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by heads"
        self.h, self.dh = heads, dim // heads
        self.w_src, self.w_dst = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.w_edge = nn.Linear(dim, dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.att)
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_emb):
        N = x.shape[0]
        src, dst = edge_index
        msg = (self.w_src(x)[src] + self.w_edge(edge_emb)).view(-1, self.h, self.dh)
        z = F.leaky_relu(self.w_dst(x)[dst].view(-1, self.h, self.dh) + msg, 0.2)
        alpha = scatter_softmax((z * self.att).sum(-1), dst, N).to(msg.dtype)        # [E, H]
        agg = torch.zeros(N, self.h, self.dh, device=x.device, dtype=msg.dtype)
        agg = agg.index_add_(0, dst, alpha.unsqueeze(-1) * msg)                      # isolated nodes -> 0
        x = self.norm1(x + self.drop(self.out(agg.reshape(N, -1))))
        return self.norm2(x + self.drop(self.ffn(x)))


class EdgeAwareGNN(nn.Module):
    """x_text [N, text_dim] + typed directed edges -> node features [N, hidden]."""

    def __init__(self, text_dim, hidden=256, heads=4, layers=3, dropout=0.0, num_relations=len(RELATIONS)):
        super().__init__()
        self.node_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden),
                                     nn.GELU(), nn.Linear(hidden, hidden))
        self.rel_emb = nn.Embedding(num_relations, hidden)      # learned relation representation
        self.layers = nn.ModuleList(EdgeGATv2Layer(hidden, heads, dropout) for _ in range(layers))
        self.hidden = hidden

    def forward(self, x_text, edge_index, edge_type):
        x = self.node_in(x_text)
        e = self.rel_emb(edge_type)
        for layer in self.layers:
            x = layer(x, edge_index, e)
        return x
