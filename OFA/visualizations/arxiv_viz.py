"""
ogbn-arxiv dataset visualization
  Visual 1 — Graph sample + 40 category bar chart
  Visual 2 — 3 task types with NOI: e2e_node, lr_node, logic_e2e
  Visual 3 — Logic task explained: OR / NOT-AND class node construction
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

import torch
import pandas as pd
import random; random.seed(42); np.random.seed(42)

print("=== arxiv ===")
from ogb.nodeproppred import PygNodePropPredDataset
ds = PygNodePropPredDataset("ogbn-arxiv",
                            root=os.path.join(OFA, "cache_data/arxiv/ST"))
data       = ds._data
edge_index = data.edge_index
num_nodes  = data.num_nodes
labels     = data.y.view(-1).numpy()

# load category names
nodeidx2paperid = pd.read_csv(
    os.path.join(OFA, "cache_data/arxiv/ST/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"),
    index_col="node idx")
cat_file = os.path.join(OFA, "data/single_graph/arxiv/arxiv_CS_categories.txt")
cat_names = {}
with open(cat_file) as f:
    lines = f.readlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("cs."):
        parts = line.split(" ", 1)
        code = parts[0].replace(".", " ").lower()
        name = parts[1][1:-1] if len(parts) > 1 else code
        cat_names[code] = name
    i += 1

label2arxiv = pd.read_csv(
    os.path.join(OFA, "data/single_graph/arxiv/labelidx2arxivcategeory.csv.gz"))
idx2name = {}
for _, row in label2arxiv.iterrows():
    code = ("arxiv " + " ".join(row["arxiv category"].strip().split(".")).lower())
    idx2name[int(row["label idx"])] = cat_names.get(code, row["arxiv category"])

print(f"  Nodes: {num_nodes}  Edges: {edge_index.shape[1]}  Classes: {len(idx2name)}")

# ── Visual 1: Graph sample + 40-class bar chart ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7),
                          gridspec_kw={"width_ratios": [1, 2]})
fig.suptitle("ogbn-arxiv — CS Paper Citation Graph  |  169,343 nodes, 40 categories",
             fontsize=13, fontweight="bold")

# 1a: graph sample
ax = axes[0]
keep_n = 300
keep   = set(random.sample(range(num_nodes), keep_n))
src, dst = edge_index[0].tolist(), edge_index[1].tolist()
G = nx.DiGraph()
G.add_edges_from([(s,d) for s,d in zip(src,dst) if s in keep and d in keep])
G.remove_edges_from(nx.selfloop_edges(G))
cmap = plt.cm.tab20
node_colors = [cmap(int(labels[n]) / 40) if n < len(labels) else "#aaa" for n in G.nodes()]
pos = nx.spring_layout(G, seed=42, k=0.5)
nx.draw_networkx(G, pos, ax=ax, with_labels=False, node_size=15,
                 node_color=node_colors, edge_color="#ccc", width=0.3,
                 alpha=0.8, arrows=True, arrowsize=4)
ax.set_title(f"300 node sample (directed citations)\nColor = CS category (40 colors)", fontsize=9)
ax.axis("off")

# 1b: 40-category bar
ax = axes[1]
unique, counts = np.unique(labels, return_counts=True)
cat_labels = [idx2name.get(int(u), str(u))[:18] for u in unique]
colors = [cmap(i/40) for i in range(len(unique))]
bars = ax.barh(cat_labels, counts, color=colors, edgecolor="white", height=0.8)
ax.set_xlabel("# papers", fontsize=9)
ax.set_title("Paper count per CS arXiv category (40 total)", fontsize=9)
ax.tick_params(axis='y', labelsize=7)
for bar, c in zip(bars, counts):
    ax.text(c + 50, bar.get_y() + bar.get_height()/2,
            str(c), va="center", fontsize=6)
ax.invert_yaxis()

save(fig, "arxiv_1_graph_categories.png")

# ── Visual 2: 3 task NOI constructions ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("arxiv — 3 Task Types with NOI Subgraph Construction",
             fontsize=12, fontweight="bold")

target     = int(np.where(labels == 0)[0][0])
G_sub, fn  = build_noi_subgraph(edge_index, num_nodes, target, hop=2, max_nodes=10)
feat_nodes = [n for n in fn if n != target][:8]
noi_id     = max(feat_nodes) + 1000
class_ids_40 = [noi_id + 1 + i for i in range(5)]  # show 5 representative
logic_ids    = [noi_id + 1 + i for i in range(4)]

draw_noi_graph(axes[0], feat_nodes, noi_id, class_ids_40, list(G_sub.edges()),
    title="e2e_node\nNode Classification (40 classes)",
    task_desc="Which CS category is this paper?\n40 class nodes connected to NOI\nE.g. cs.AI, cs.LG, cs.CV...",
    class_labels=list(idx2name.values())[:5])

draw_noi_graph(axes[1], feat_nodes, noi_id, [], list(G_sub.edges()),
    title="lr_node\nFew-Shot Classification",
    task_desc="Classify using only k labeled\nexamples per class.\nNo class nodes in subgraph.\nComparison done by GNN similarity.")

draw_noi_graph(axes[2], feat_nodes[:8], noi_id, logic_ids, list(G_sub.edges()),
    title="logic_e2e\nLogic Reasoning",
    task_desc="Does paper match compound logic?\ne.g. 'cs.AI OR cs.LO'\n      'NOT cs.DB AND NOT cs.NI'\n4 logic class nodes",
    class_labels=["A OR B", "C OR D", "NOT A\nNOT B", "NOT C\nNOT D"])

save(fig, "arxiv_2_task_types.png")

# ── Visual 3: Logic task deep-dive ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("arxiv logic_e2e — How OR and NOT-AND Class Nodes Are Built",
             fontsize=12, fontweight="bold")

ax = axes[0]; ax.axis("off"); ax.set_facecolor("#FFF8E1")
eg_cats = list(idx2name.values())[:6]
or_text = "OR CLASS NODES (either A or B):\n\n"
for i in range(min(3, len(eg_cats))):
    for j in range(i+1, min(4, len(eg_cats))):
        or_text += f"  'either {eg_cats[i][:20]} or {eg_cats[j][:20]}'\n"
or_text += f"\nTotal OR nodes = 40 x 40 = 1,600"
ax.text(0.05, 0.95, or_text, transform=ax.transAxes, fontsize=8,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("OR class nodes", fontsize=10, fontweight="bold")

ax = axes[1]; ax.axis("off"); ax.set_facecolor("#FCE4EC")
and_text = "NOT-AND CLASS NODES (neither A nor B):\n\n"
for i in range(min(3, len(eg_cats))):
    for j in range(i+1, min(4, len(eg_cats))):
        and_text += f"  'not {eg_cats[i][:20]} and not {eg_cats[j][:20]}'\n"
and_text += f"\nTotal NOT-AND nodes = 40 x 40 = 1,600\n"
and_text += "\nTotal logic class nodes = 3,200\n(appended to the 40 regular classes)"
ax.text(0.05, 0.95, and_text, transform=ax.transAxes, fontsize=8,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("NOT-AND class nodes", fontsize=10, fontweight="bold")

save(fig, "arxiv_3_logic_task.png")
print("arxiv done.\n")
