"""
Cora dataset visualization
  Visual 1 — Full graph structure (5% sample) + class distribution
  Visual 2 — All 4 task types with NOI subgraph construction:
             e2e_node, lr_node, e2e_link, logic_e2e
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

import torch
import pandas as pd
import matplotlib.gridspec as gridspec
import random; random.seed(42); np.random.seed(42)

print("=== Cora ===")
data = torch.load(os.path.join(OFA, "data/single_graph/Cora/cora.pt"))
cat_csv = pd.read_csv(os.path.join(OFA, "data/single_graph/Cora/categories.csv"), sep=",")

num_nodes   = data.num_nodes
edge_index  = data.edge_index
labels      = data.y.numpy() if hasattr(data.y, "numpy") else data.y
label_names = data.label_names
texts       = data.raw_texts

print(f"  Nodes: {num_nodes}  Edges: {edge_index.shape[1]//2}  Classes: {len(label_names)}")
print(f"  Classes: {label_names}")

# ── Visual 1: Graph + class distribution ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Cora — Citation Graph  |  2,708 papers, 7 CS categories", fontsize=13, fontweight="bold")

# 1a: 5% subgraph
ax = axes[0]
keep_n = max(100, int(num_nodes * 0.05))
keep   = set(random.sample(range(num_nodes), keep_n))
src, dst = edge_index[0].tolist(), edge_index[1].tolist()
edges_sub = [(s,d) for s,d in zip(src,dst) if s in keep and d in keep]
G = nx.Graph(); G.add_edges_from(edges_sub)
G.remove_edges_from(nx.selfloop_edges(G))
colors7 = ["#E53935","#8E24AA","#1E88E5","#43A047","#FB8C00","#00ACC1","#6D4C41"]
node_colors = [colors7[int(labels[n]) % 7] if n < len(labels) else "#aaa"
               for n in G.nodes()]
pos = nx.spring_layout(G, seed=42, k=0.8)
nx.draw_networkx(G, pos, ax=ax, with_labels=False, node_size=25,
                 node_color=node_colors, edge_color="#ccc", width=0.5, alpha=0.85)
legend_h = [mpatches.Patch(color=colors7[i], label=str(label_names[i])[:20])
            for i in range(len(label_names))]
ax.legend(handles=legend_h, fontsize=6, loc="lower left", ncol=1)
ax.set_title(f"5% subgraph ({keep_n} nodes)\nColored by class", fontsize=9)
ax.axis("off")

# 1b: class distribution
ax = axes[1]
unique, counts = np.unique(labels, return_counts=True)
ax.bar([str(label_names[i])[:15] for i in unique], counts,
       color=[colors7[i] for i in unique], edgecolor="white")
ax.set_title("Node class distribution\n(how many papers per category)", fontsize=9)
ax.set_xlabel("Category", fontsize=8); ax.set_ylabel("# papers", fontsize=8)
ax.tick_params(axis='x', rotation=35, labelsize=7)
for i, c in enumerate(counts):
    ax.text(i, c+5, str(c), ha="center", fontsize=7)

# 1c: example paper node text
ax = axes[2]
ax.axis("off")
ax.set_facecolor("#F5F5F5")
examples = []
for cls_idx, cls_name in enumerate(label_names):
    node_idx = int(np.where(labels == cls_idx)[0][0])
    txt = str(texts[node_idx])[:120] + "..."
    examples.append(f"Class {cls_idx} — {cls_name}:\n  \"{txt}\"\n")
full_text = "REAL NODE TEXT EXAMPLES (what goes into Sentence-BERT)\n\n" + "\n".join(examples)
ax.text(0.02, 0.98, full_text, transform=ax.transAxes, fontsize=6.2,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("What node features look like\n(raw text → Sentence-BERT → 768-dim)", fontsize=9)

save(fig, "cora_1_graph_classes.png")

# ── Visual 2: 4 task NOI constructions ───────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 7))
fig.suptitle("Cora — 4 Task Types: How Subgraphs Are Built with NOI Node",
             fontsize=12, fontweight="bold")

# pick a target node
target = int(np.where(labels == 0)[0][5])
G_sub, feat_nodes = build_noi_subgraph(edge_index, num_nodes, target, hop=2, max_nodes=12)
feat_nodes = [n for n in feat_nodes if n != target][:10]
noi_id = max(feat_nodes) + 1000
class_ids = [noi_id + 1 + i for i in range(len(label_names))]
class_ids_link = [noi_id + 1, noi_id + 2]       # binary: co-cited / not
logic_ids = [noi_id + 1 + i for i in range(4)]  # 4 logic combinations

sub_edges = list(G_sub.edges())

# Task 1: e2e_node
draw_noi_graph(axes[0], feat_nodes, noi_id, class_ids, sub_edges,
    title="e2e_node\nEnd-to-end Node Classification",
    task_desc="Q: Which of 7 CS categories\n   does target paper belong to?\nNOI→ClassNodes: all 7 classes\nEdges: f↔NOI↔classes",
    class_labels=[str(n)[:8] for n in label_names])

# Task 2: lr_node (few-shot — no class nodes in graph)
draw_noi_graph(axes[1], feat_nodes, noi_id, [], sub_edges,
    title="lr_node\nFew-Shot Node Classification",
    task_desc="Q: Classify with only a few\n   labeled examples (k-shot)\nNO class nodes attached\nSupport graphs compared separately")

# Task 3: e2e_link
draw_noi_graph(axes[2], feat_nodes[:6], noi_id, class_ids_link, sub_edges[:8],
    title="e2e_link\nLink Prediction",
    task_desc="Q: Are these two papers\n   co-cited? (yes/no)\n2 class nodes: co-cited / not\nTarget = edge between papers",
    class_labels=["not co-cited", "co-cited"])

# Task 4: logic_e2e
draw_noi_graph(axes[3], feat_nodes[:8], noi_id, logic_ids, sub_edges[:10],
    title="logic_e2e\nLogic Reasoning",
    task_desc="Q: Does paper belong to\n   'X OR Y' / 'NOT X AND NOT Y'?\n4 logic combos as class nodes\ne.g. 'cs.AI OR cs.LO'",
    class_labels=["A OR B","C OR D","NOT A\nNOT B","NOT C\nNOT D"])

save(fig, "cora_2_task_types.png")
print("Cora done.\n")
