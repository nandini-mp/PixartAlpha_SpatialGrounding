"""Batch several LayoutGraphs into one big disjoint graph (node offsets, no padding)."""
import torch

from src.graph.relations import REL2ID


def batch_graphs(graphs, embedder):
    """Returns dict with:
       x_text [N, D]     frozen description embeddings
       edge_index [2, E] (src, dst); edge (s, d, REL) means "s REL d"; both directions present
       edge_type [E]     relation ids
       node_graph [N]    graph index of each node
       offsets [B]       first node index of each graph
    """
    xs, ei, et, ng, offs, off = [], [], [], [], [], 0
    for gi, g in enumerate(graphs):
        xs.append(embedder.get(g.descs))
        for s, d, rel in g.edges:
            ei.append((s + off, d + off))
            et.append(REL2ID[rel])
        ng += [gi] * g.n
        offs.append(off)
        off += g.n
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.zeros(2, 0, dtype=torch.long)
    return {"x_text": torch.cat(xs, 0), "edge_index": edge_index,
            "edge_type": torch.tensor(et, dtype=torch.long),
            "node_graph": torch.tensor(ng, dtype=torch.long),
            "offsets": torch.tensor(offs, dtype=torch.long)}
