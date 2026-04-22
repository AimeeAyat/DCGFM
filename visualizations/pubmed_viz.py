"""
Pubmed dataset visualization
  Visual 1 — Graph structure + 3 class distribution
  Visual 2 — 3 task types with NOI: e2e_node, lr_node, e2e_link
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

import torch
import matplotlib.gridspec as gridspec
import random; random.seed(42); np.random.seed(42)

print("=== Pubmed ===")
data = torch.load(os.path.join(OFA, "data/single_graph/Pubmed/pubmed.pt"))

num_nodes  = data.num_nodes
edge_index = data.edge_index
labels     = data.y.numpy() if hasattr(data.y, "numpy") else data.y
texts      = data.raw_texts

with open(os.path.join(OFA, "data/single_graph/Pubmed/categories.csv")) as f:
    class_names = [l.strip()[:50] for l in f.read().split("\n") if l.strip()]

print(f"  Nodes: {num_nodes}  Edges: {edge_index.shape[1]//2}  Classes: {len(class_names)}")
print(f"  Classes: {[c[:30] for c in class_names]}")

colors3 = ["#E53935", "#1E88E5", "#43A047"]

# ── Visual 1: Graph + class dist + node text examples ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Pubmed — Medical Citation Graph  |  19,717 papers, 3 Diabetes classes",
             fontsize=13, fontweight="bold")

# 1a: graph sample
ax = axes[0]
keep_n = max(150, int(num_nodes * 0.01))
keep   = set(random.sample(range(num_nodes), keep_n))
src, dst = edge_index[0].tolist(), edge_index[1].tolist()
G = nx.Graph()
G.add_edges_from([(s,d) for s,d in zip(src,dst) if s in keep and d in keep])
G.remove_edges_from(nx.selfloop_edges(G))
node_colors = [colors3[int(labels[n]) % 3] if n < len(labels) else "#aaa"
               for n in G.nodes()]
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, pos, ax=ax, with_labels=False, node_size=30,
                 node_color=node_colors, edge_color="#ccc", width=0.5, alpha=0.85)
short = ["Exp.", "Type 1", "Type 2"]
legend_h = [mpatches.Patch(color=colors3[i], label=short[i]) for i in range(3)]
ax.legend(handles=legend_h, fontsize=8)
ax.set_title(f"1% sample ({keep_n} nodes)\nColored by Diabetes type", fontsize=9)
ax.axis("off")

# 1b: class distribution
ax = axes[1]
unique, counts = np.unique(labels, return_counts=True)
bars = ax.bar(short, counts, color=colors3, edgecolor="white", width=0.5)
ax.set_title("Node class distribution\n(# papers per Diabetes category)", fontsize=9)
ax.set_ylabel("# papers", fontsize=8)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, c + 50, str(c), ha="center", fontsize=9)
# show percentage
total = counts.sum()
for i, c in enumerate(counts):
    ax.text(i, c/2, f"{100*c/total:.1f}%", ha="center", fontsize=10,
            color="white", fontweight="bold")

# 1c: class name + example text
ax = axes[2]; ax.axis("off"); ax.set_facecolor("#F0F4F8")
info = "PUBMED CLASS DESCRIPTIONS & EXAMPLE NODES\n\n"
for cls_idx, cls_name in enumerate(class_names):
    node_idx = int(np.where(labels == cls_idx)[0][0])
    info += f"Class {cls_idx}: {cls_name[:45]}...\n"
    info += f"  e.g. \"{str(texts[node_idx])[:90]}...\"\n\n"
ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=6.5,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("Class descriptions + example paper text", fontsize=9)

save(fig, "pubmed_1_graph_classes.png")

# ── Visual 2: 3 task NOI constructions ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("Pubmed — 3 Task Types with NOI Subgraph Construction",
             fontsize=12, fontweight="bold")

target     = int(np.where(labels == 0)[0][3])
G_sub, fn  = build_noi_subgraph(edge_index, num_nodes, target, hop=2, max_nodes=10)
feat_nodes = [n for n in fn if n != target][:8]
noi_id     = max(feat_nodes) + 1000
class_ids  = [noi_id + 1 + i for i in range(3)]
link_ids   = [noi_id + 1, noi_id + 2]

draw_noi_graph(axes[0], feat_nodes, noi_id, class_ids, list(G_sub.edges()),
    title="e2e_node\nNode Classification",
    task_desc="Which Diabetes type is this paper?\nExp / Type1 / Type2\nAll 3 class nodes attached to NOI",
    class_labels=["Exp.", "Type 1", "Type 2"])

draw_noi_graph(axes[1], feat_nodes, noi_id, [], list(G_sub.edges()),
    title="lr_node\nFew-Shot Node Classification",
    task_desc="Classify with few examples.\nNo class nodes in graph.\nSupport set compared externally.")

draw_noi_graph(axes[2], feat_nodes[:6], noi_id, link_ids,
    [(s,d) for s,d in G_sub.edges()][:8],
    title="e2e_link\nLink / Co-Citation Prediction",
    task_desc="Are papers A and B co-cited?\nBinary: yes(1) or no(0)\n2 class nodes: not-cocited / cocited",
    class_labels=["not co-cited", "co-cited"])

save(fig, "pubmed_2_task_types.png")
print("Pubmed done.\n")
