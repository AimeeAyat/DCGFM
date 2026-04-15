"""
WikiCS dataset visualization
  Visual 1 — Graph sample + 10 category distribution
  Visual 2 — 2 task types with NOI: e2e_node, lr_node
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

import torch, json
import random; random.seed(42); np.random.seed(42)

print("=== WikiCS ===")
from torch_geometric.datasets import WikiCS
wikics_path = os.path.join(OFA, "data/single_graph/wikics")
ds   = WikiCS(root=wikics_path)
data = ds._data
edge_index = data.edge_index
num_nodes  = data.num_nodes
labels     = data.y.numpy() if hasattr(data.y, "numpy") else data.y

with open(os.path.join(wikics_path, "metadata.json")) as f:
    meta = json.load(f)
label_names = list(meta["labels"].values())   # 10 CS Wikipedia categories
node_titles = [meta["nodes"][i]["title"] for i in range(min(num_nodes, len(meta["nodes"])))]

print(f"  Nodes: {num_nodes}  Edges: {edge_index.shape[1]//2}  Classes: {len(label_names)}")
print(f"  Categories: {label_names}")

colors10 = plt.cm.tab10(np.linspace(0, 1, 10))

# ── Visual 1: Graph + class dist + example titles ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("WikiCS — Wikipedia CS Articles  |  11,701 nodes, 10 categories",
             fontsize=13, fontweight="bold")

# 1a graph
ax = axes[0]
keep_n = 200
keep   = set(random.sample(range(num_nodes), keep_n))
src, dst = edge_index[0].tolist(), edge_index[1].tolist()
G = nx.Graph()
G.add_edges_from([(s,d) for s,d in zip(src,dst) if s in keep and d in keep])
G.remove_edges_from(nx.selfloop_edges(G))
node_colors = [colors10[int(labels[n])] if n < len(labels) else [0.5]*4 for n in G.nodes()]
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, pos, ax=ax, with_labels=False, node_size=25,
                 node_color=node_colors, edge_color="#ccc", width=0.5, alpha=0.85)
legend_h = [mpatches.Patch(color=colors10[i], label=label_names[i][:20])
            for i in range(len(label_names))]
ax.legend(handles=legend_h, fontsize=6, loc="lower left")
ax.set_title(f"Graph sample ({keep_n} nodes)\nColored by Wikipedia category", fontsize=9)
ax.axis("off")

# 1b class dist
ax = axes[1]
unique, counts = np.unique(labels, return_counts=True)
ax.barh([label_names[i][:25] for i in unique], counts,
        color=[colors10[i] for i in unique], edgecolor="white")
ax.set_xlabel("# Wikipedia articles", fontsize=8)
ax.set_title("Article count per category", fontsize=9)
ax.tick_params(axis='y', labelsize=7)
for i, c in enumerate(counts):
    ax.text(c+5, i, str(c), va="center", fontsize=7)
ax.invert_yaxis()

# 1c example node texts
ax = axes[2]; ax.axis("off"); ax.set_facecolor("#F0F4F8")
ex_text = "REAL NODE TEXT EXAMPLES\n(Wikipedia title + tokens → Sentence-BERT)\n\n"
for cls_idx in range(len(label_names)):
    node_idx = int(np.where(labels == cls_idx)[0][0])
    title    = node_titles[node_idx] if node_idx < len(node_titles) else "N/A"
    tokens   = " ".join(meta["nodes"][node_idx]["tokens"][:12]) + "..."
    ex_text += f"[{label_names[cls_idx][:20]}]\n"
    ex_text += f"  Title: {title[:40]}\n"
    ex_text += f"  Text:  {tokens[:60]}\n\n"
ax.text(0.02, 0.98, ex_text, transform=ax.transAxes, fontsize=6,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("Node feature = Wikipedia article text", fontsize=9)

save(fig, "wikics_1_graph_categories.png")

# ── Visual 2: 2 task NOI constructions ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("WikiCS — 2 Task Types with NOI Subgraph Construction",
             fontsize=12, fontweight="bold")

target     = int(np.where(labels == 0)[0][2])
G_sub, fn  = build_noi_subgraph(edge_index, num_nodes, target, hop=2, max_nodes=10)
feat_nodes = [n for n in fn if n != target][:8]
noi_id     = max(feat_nodes) + 1000
class_ids  = [noi_id + 1 + i for i in range(len(label_names))]

draw_noi_graph(axes[0], feat_nodes, noi_id, class_ids, list(G_sub.edges()),
    title="e2e_node\nEnd-to-End Node Classification",
    task_desc="Which CS topic is this Wikipedia article?\n10 class nodes (categories) attached.\nNOI connects to all feature + class nodes.",
    class_labels=[l[:10] for l in label_names])

draw_noi_graph(axes[1], feat_nodes, noi_id, [], list(G_sub.edges()),
    title="lr_node\nFew-Shot Node Classification",
    task_desc="Classify with only few labeled examples.\nNo class nodes in graph — comparison\ndone by embedding similarity in GNN.")

save(fig, "wikics_2_task_types.png")
print("WikiCS done.\n")
