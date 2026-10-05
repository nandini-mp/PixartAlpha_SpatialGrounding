"""Stage 5: structure -> layout graph (typed directed edges, components, anchors, BFS order).

Node ids are 0-based indices in the order of struct["objects"].
All binary relations are stored in BOTH directions (relation + its inverse), so traversal
from any parent sees the relation from the parent's perspective:
    child_rel = relation such that "child REL parent" holds.
Concretely, edge (src, dst, rel) means "src REL dst".
"""
from collections import deque
from dataclasses import dataclass, field

from src.graph.relations import INVERSE, REL2ID


@dataclass
class LayoutGraph:
    descs: list                       # node descriptions (str)
    edges: list                       # (src, dst, REL) both directions, deduplicated, sorted
    betweens: list                    # (mid, a, b, axis) ternary constraints
    ids: list                         # original structure ids (for mapping back)
    adj: dict = field(default_factory=dict)       # node -> sorted neighbor list
    components: list = field(default_factory=list)  # list of sorted node lists
    anchors: list = field(default_factory=list)     # anchor node per component
    bfs: list = field(default_factory=list)         # per component: list of (node, parent, rel, depth)
    comp_of: list = field(default_factory=list)     # node -> component index
    warnings: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.descs)

    def depth_of(self):
        d = [0] * self.n
        for comp in self.bfs:
            for node, _, _, dep in comp:
                d[node] = dep
        return d

    def parent_of(self):
        p = [-1] * self.n
        for comp in self.bfs:
            for node, par, _, _ in comp:
                p[node] = par
        return p


def build_graph(struct):
    """struct: {"objects":[{"id","desc"}], "relations":[{"rel","args"}]} -> LayoutGraph."""
    objs = struct["objects"]
    if not objs:
        raise ValueError("cannot build a graph with zero objects")
    ids = [o["id"] for o in objs]
    idx = {oid: i for i, oid in enumerate(ids)}
    if len(idx) != len(ids):
        raise ValueError("duplicate object ids")
    descs = [o["desc"] for o in objs]

    edge_set, betweens, warnings = set(), [], []

    def add(s, d, rel):
        if s == d:
            return
        edge_set.add((s, d, rel))
        edge_set.add((d, s, INVERSE[rel]))

    for r in struct["relations"]:
        rel, args = r["rel"].upper(), [idx[a] for a in r["args"]]
        if rel in ("BETWEEN_X", "BETWEEN_Y"):
            mid, a, b = args
            betweens.append((mid, a, b, rel[-1].lower()))
            add(mid, a, rel)
            add(mid, b, rel)
        else:
            if rel not in REL2ID:
                raise ValueError(f"unknown relation {rel}")
            add(args[0], args[1], rel)

    edges = sorted(edge_set)
    adj = {i: set() for i in range(len(objs))}
    for s, d, _ in edges:
        adj[s].add(d)
    adj = {k: sorted(v) for k, v in adj.items()}

    # connected components (undirected, since every edge exists both ways)
    comp_of, components = [-1] * len(objs), []
    for start in range(len(objs)):
        if comp_of[start] != -1:
            continue
        comp, q = [], deque([start])
        comp_of[start] = len(components)
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if comp_of[v] == -1:
                    comp_of[v] = len(components); q.append(v)
        components.append(sorted(comp))

    # anchor = highest degree (unique neighbors), tie -> smallest index
    anchors = [min(c, key=lambda u: (-len(adj[u]), u)) for c in components]

    # BFS order from each anchor; first stored edge (sorted) per (parent, child) gives the relation
    rel_of = {}
    for s, d, rel in edges:
        rel_of.setdefault((s, d), []).append(rel)
    bfs = []
    for comp, anc in zip(components, anchors):
        order, seen, q = [(anc, -1, None, 0)], {anc}, deque([(anc, 0)])
        while q:
            u, du = q.popleft()
            for v in adj[u]:                       # sorted -> deterministic
                if v in seen:
                    continue
                seen.add(v)
                # relation such that "v REL u" holds = inverse of the edge stored as (u -> v)
                rel_u_to_v = rel_of[(u, v)][0]
                order.append((v, u, INVERSE[rel_u_to_v], du + 1))
                q.append((v, du + 1))
        bfs.append(order)

    g = LayoutGraph(descs=descs, edges=edges, betweens=betweens, ids=ids, adj=adj,
                    components=components, anchors=anchors, bfs=bfs, comp_of=comp_of,
                    warnings=warnings)
    g.warnings = validate_graph(g)
    return g


def validate_graph(g):
    """Structural checks (raise AssertionError-style problems as strings). Returns warning list.
    Hard structural failures raise ValueError."""
    warns = []
    covered = []
    for comp, anc, order in zip(g.components, g.anchors, g.bfs):
        nodes = [x[0] for x in order]
        if sorted(nodes) != comp:
            raise ValueError("BFS did not cover its component exactly once")
        if order[0][0] != anc or order[0][1] != -1 or order[0][3] != 0:
            raise ValueError("first BFS entry must be the anchor with no parent")
        pos = {n: i for i, n in enumerate(nodes)}
        dep = {n: d for n, _, _, d in order}
        for node, par, rel, d in order[1:]:
            if par not in pos or pos[par] >= pos[node]:
                raise ValueError("parent must precede child in BFS order")
            if d != dep[par] + 1:
                raise ValueError("depth inconsistency")
            if (node, par, rel) not in set(g.edges):
                raise ValueError("BFS relation not backed by a graph edge")
        covered += nodes
    if sorted(covered) != list(range(g.n)):
        raise ValueError("some node did not receive a place in the BFS order")

    # contradictions: same ordered pair with opposite directional relations
    by_pair = {}
    for s, d, rel in g.edges:
        by_pair.setdefault((s, d), set()).add(rel)
    for (s, d), rels in by_pair.items():
        if s < d:
            if {"LEFT", "RIGHT"} <= rels or {"ABOVE", "BELOW"} <= rels or {"INSIDE", "CONTAINS"} <= rels \
               or {"NEAR", "FAR"} <= rels:
                warns.append(f"contradictory relations between {s} and {d}: {sorted(rels)}")
    return warns


def describe_graph(g):
    """Human-readable summary (for debugging)."""
    lines = [f"nodes={g.n} components={len(g.components)}"]
    for ci, (comp, anc, order) in enumerate(zip(g.components, g.anchors, g.bfs)):
        lines.append(f" comp {ci}: nodes={comp} anchor={anc}")
        for node, par, rel, dep in order:
            lines.append(f"   {node} ({g.descs[node]}) depth={dep}"
                         + ("" if par < 0 else f" parent={par} rel_from_parent={rel}"))
    if g.betweens:
        lines.append(f" betweens (mid,a,b,axis): {g.betweens}")
    return "\n".join(lines)
