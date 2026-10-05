"""Stage 12: GT vs predicted layout figures. Boxes are normalized (cx, cy, w, h), y grows downward."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEPTH_COLORS = {0: "#d62728", 1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd"}


def _draw(ax, boxes, graph, title):
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    depth, parent = graph.depth_of(), graph.parent_of()
    for i, (cx, cy, w, h) in enumerate(boxes.tolist()):
        col = DEPTH_COLORS[min(depth[i], 3)]
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=True, alpha=0.18, color=col, lw=0))
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec=col, lw=2))
        ax.text(cx, cy, f"{i}:{graph.descs[i]}", ha="center", va="center", fontsize=7, color="black")
    for i, p in enumerate(parent):
        if p >= 0:
            ax.annotate("", xy=tuple(boxes[i, :2].tolist()), xytext=tuple(boxes[p, :2].tolist()),
                        arrowprops=dict(arrowstyle="->", ls="--", color="gray", lw=1))
    ax.set_title(title, fontsize=8)


def plot_pair(gt, pred, graph, iou, ok, path, header=""):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    _draw(axes[0], gt, graph, "ground truth")
    _draw(axes[1], pred, graph, f"prediction | IoU {iou:.2f} | {'all constraints OK' if ok else 'constraint violated'}")
    rels = "; ".join(f"{s} {r} {d}" for s, d, r in graph.edges if r in ("LEFT", "ABOVE", "INSIDE")
                     or (r in ("NEAR", "FAR", "OVERLAP") and s < d))
    bt = "; ".join(f"{m} between_{ax} {a},{b}" for m, a, b, ax in graph.betweens)
    fig.suptitle((header + " " + rels + ("; " + bt if bt else ""))[:170], fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)
