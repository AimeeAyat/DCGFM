"""
Visualization script: arxiv subgraphs → GIN hard pruning → kept vs pruned

What this shows:
  1. What the ogbn-arxiv dataset looks like (citation graph + subgraphs)
  2. What goes INTO the hard pruning module (subgraphs as PyG Data objects)
  3. What comes OUT (anomaly scores per subgraph)
  4. Visual comparison of kept vs pruned subgraphs

Run from repo root:
    python visualize_pipeline.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────
# 1.  Simulate what ogbn-arxiv subgraphs look like
#     Real pipeline: ogbn-arxiv has 169,343 nodes (papers),
#     1,166,243 edges (citations). For each training node,
#     a 2-hop subgraph is extracted.  Node features are
#     768-dim Sentence-BERT embeddings of "title + abstract".
#
#     Here we simulate three graph "types" to mimic real variety:
#       - "star"    → dense hub node  (many citations to one paper)
#       - "chain"   → linear path     (sequential citation chain)
#       - "cluster" → tight clique    (group of co-citing papers)
# ─────────────────────────────────────────────────────────────

NODE_FEAT_DIM = 32   # 768 in real pipeline (Sentence-BERT)
NUM_GRAPHS    = 40
PRUNE_RATIO   = 0.5  # keep top 50% by anomaly score


def make_star_graph(n=10, feat_dim=NODE_FEAT_DIM):
    """Hub node connected to all others – common in highly-cited papers."""
    G = nx.star_graph(n - 1)
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    x = torch.randn(n, feat_dim) * 0.5          # tight cluster in feature space
    x[0] += 3.0                                  # hub node is distinctive
    return Data(x=x, edge_index=edge_index, y=torch.tensor([0]),
                num_nodes=n, graph_type="star")


def make_chain_graph(n=10, feat_dim=NODE_FEAT_DIM):
    """Linear citation chain – niche sequential work."""
    G = nx.path_graph(n)
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    x = torch.randn(n, feat_dim) * 0.3
    for i in range(n):
        x[i] += torch.tensor([i * 0.2] + [0.0] * (feat_dim - 1))
    return Data(x=x, edge_index=edge_index, y=torch.tensor([1]),
                num_nodes=n, graph_type="chain")


def make_cluster_graph(n=10, feat_dim=NODE_FEAT_DIM):
    """Dense cluster – group of tightly related papers."""
    G = nx.erdos_renyi_graph(n, p=0.7, seed=42)
    if G.number_of_edges() == 0:
        G.add_edge(0, 1)
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    x = torch.randn(n, feat_dim) * 0.2           # very tight in feature space (normal)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([2]),
                num_nodes=n, graph_type="cluster")


def make_outlier_graph(n=10, feat_dim=NODE_FEAT_DIM):
    """Isolated/noisy subgraph – uninformative, far from typical."""
    edge_index = torch.zeros(2, 2, dtype=torch.long)  # nearly no edges
    x = torch.randn(n, feat_dim) * 3.0 + 10.0         # far from origin → high anomaly
    return Data(x=x, edge_index=edge_index, y=torch.tensor([3]),
                num_nodes=n, graph_type="outlier")


# Build a mixed dataset: mostly normal, a few outliers
data_list = []
types = []
for i in range(NUM_GRAPHS):
    r = i % 10
    if r < 4:
        g = make_cluster_graph(n=random.randint(6, 14))
        types.append("cluster")
    elif r < 7:
        g = make_star_graph(n=random.randint(6, 12))
        types.append("star")
    elif r < 9:
        g = make_chain_graph(n=random.randint(5, 12))
        types.append("chain")
    else:
        g = make_outlier_graph(n=random.randint(5, 10))
        types.append("outlier")
    data_list.append(g)

print(f"Dataset: {NUM_GRAPHS} subgraphs")
print(f"  cluster: {types.count('cluster')}, star: {types.count('star')}, "
      f"chain: {types.count('chain')}, outlier: {types.count('outlier')}")
print(f"  Node feature dim: {NODE_FEAT_DIM}  (real pipeline: 768 from Sentence-BERT)")
print(f"  Example graph: nodes={data_list[0].num_nodes}, "
      f"edges={data_list[0].edge_index.shape[1]//2}\n")


# ─────────────────────────────────────────────────────────────
# 2.  Run the hard pruning module
#     Input:  data_list  – list of PyG Data (subgraphs)
#     Output: anomaly_scores – float array, one score per graph
# ─────────────────────────────────────────────────────────────

import sys, os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
import pytorch_lightning as pl

# ── Inline hard_prune_api (avoids importing OFA/hard_prune_module.py which has
#    module-level code that crashes on import) ──────────────────────────────────

class _GIN(nn.Module):
    def __init__(self, nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False):
        super().__init__()
        self.norm = BatchNorm1d
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.nns = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(nlayer):
            self.nns.append(Sequential(Linear(nhid, nhid, bias=bias), act, Linear(nhid, nhid, bias=bias)))
            self.convs.append(GINConv(self.nns[-1]))
            self.bns.append(self.norm(nhid))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.transform(x)
        embed = self.pooling(x, batch)
        std = torch.sqrt(self.pooling((x - embed[batch]) ** 2, batch))
        graph_embeds, graph_stds = [embed], [std]
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)
            embed = self.pooling(x, batch)
            std = torch.sqrt(self.pooling((x - embed[batch]) ** 2, batch))
            graph_embeds.append(embed)
            graph_stds.append(std)
        return torch.stack(graph_embeds), torch.stack(graph_stds)


class _GIN_Hard_Prune(pl.LightningModule):
    def __init__(self, nfeat, nhid=128, nlayer=3, dropout=0, learning_rate=0.001, weight_decay=0):
        super().__init__()
        self.model = _GIN(nfeat, nhid, nlayer=nlayer, dropout=dropout)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.radius = 0
        self.nu = 1
        self.mode = 'sum'
        self.register_buffer('center', torch.zeros(nhid))
        self.register_buffer('all_layer_centers', torch.zeros(nlayer + 1, nhid))
        self.training_step_outputs = []
        self.test_step_outputs = []

    def forward(self, data):
        embs, _ = self.model(data)
        embs = embs.sum(dim=0)
        dist = torch.sum((embs - self.center) ** 2, dim=1)
        return dist - self.radius ** 2

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            embs, _ = self.model(batch)
            loss = torch.zeros(1, requires_grad=True, device=self.device)
            self.training_step_outputs.append({'loss': loss, 'emb': embs.detach()})
            return {'loss': loss, 'emb': embs.detach()}
        else:
            scores = self(batch)
            loss = self.radius ** 2 + (1 / self.nu) * torch.mean(F.relu(scores))
            self.training_step_outputs.append(loss)
            return loss

    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            embs = torch.cat([d['emb'] for d in self.training_step_outputs], dim=1)
            self.all_layer_centers = embs.mean(dim=1)
            self.center = torch.sum(self.all_layer_centers, 0)
        self.training_step_outputs = []

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self(batch))
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


class _SimpleDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)
    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return []
    def download(self): pass
    def process(self): pass


def hard_prune_api(data_list, batch_size=32, weight_decay=5e-4, nlayer=5, max_epochs=25, devices=[0]):
    dataset = _SimpleDataset(data_list)
    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = _GIN_Hard_Prune(dataset[0].x.shape[1], weight_decay=weight_decay, nlayer=nlayer)
    trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=max_epochs, logger=False)
    trainer.fit(model=model, train_dataloaders=train_dl)
    trainer.test(model=model, dataloaders=test_dl)
    local_scores = torch.cat([out for out in model.test_step_outputs])
    return local_scores.cpu().detach().numpy()

print("Running hard pruning (GIN + Deep SVDD)...")
anomaly_scores = hard_prune_api(
    data_list, batch_size=16, nlayer=3, max_epochs=15, devices=[0]
)
print(f"\nAnomaly scores (one per graph):\n{np.round(anomaly_scores, 2)}\n")


# ─────────────────────────────────────────────────────────────
# 3.  Apply pruning decision
# ─────────────────────────────────────────────────────────────

threshold = np.percentile(anomaly_scores, PRUNE_RATIO * 100)
# --hard_pruning_reverse: keep HIGH scores (harder samples)
kept_mask   = anomaly_scores >= threshold
pruned_mask = ~kept_mask

kept_idx   = np.where(kept_mask)[0]
pruned_idx = np.where(pruned_mask)[0]

print(f"Threshold (50th percentile): {threshold:.2f}")
print(f"Kept   ({len(kept_idx)} graphs): scores >= {threshold:.2f}")
print(f"Pruned ({len(pruned_idx)} graphs): scores <  {threshold:.2f}")


# ─────────────────────────────────────────────────────────────
# 4.  Visualize
# ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    "DCGFM Hard Pruning Pipeline\n"
    "ogbn-arxiv subgraphs → GIN (Deep SVDD) → anomaly scores → keep / prune",
    fontsize=14, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

# ── Panel A: example subgraphs ──────────────────────────────
gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.3)
example_indices = [
    next(i for i, t in enumerate(types) if t == "cluster"),
    next(i for i, t in enumerate(types) if t == "star"),
    next(i for i, t in enumerate(types) if t == "chain"),
    next(i for i, t in enumerate(types) if t == "outlier"),
]
titles_ex = ["Cluster subgraph\n(tight co-citation)",
             "Star subgraph\n(highly-cited paper)",
             "Chain subgraph\n(sequential citations)",
             "Outlier subgraph\n(noisy / isolated)"]
colors_ex = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

for col, (idx, title, color) in enumerate(zip(example_indices, titles_ex, colors_ex)):
    ax = fig.add_subplot(gs_top[col])
    g = data_list[idx]
    G = to_networkx(g, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, ax=ax, with_labels=False,
                     node_color=color, node_size=120, edge_color="#555",
                     width=1.2, alpha=0.85)
    ax.set_title(f"{title}\n{g.num_nodes} nodes, {G.number_of_edges()} edges",
                 fontsize=9)
    ax.axis("off")
    score_str = f"score={anomaly_scores[idx]:.1f}"
    ax.text(0.5, -0.08, score_str, ha="center", transform=ax.transAxes,
            fontsize=9, color=color, fontweight="bold")

fig.text(0.5, 0.685, "Panel A — What the input subgraphs look like (node = paper, edge = citation)",
         ha="center", fontsize=10, style="italic")

# ── Panel B: anomaly score distribution ─────────────────────
ax_bar = fig.add_subplot(gs[1])
bar_colors = []
for i in range(NUM_GRAPHS):
    if types[i] == "outlier":
        bar_colors.append("#F44336")
    elif types[i] == "cluster":
        bar_colors.append("#4CAF50")
    elif types[i] == "star":
        bar_colors.append("#2196F3")
    else:
        bar_colors.append("#FF9800")

sorted_idx = np.argsort(anomaly_scores)
sorted_scores = anomaly_scores[sorted_idx]
sorted_colors = [bar_colors[i] for i in sorted_idx]
sorted_kept   = kept_mask[sorted_idx]

bars = ax_bar.bar(range(NUM_GRAPHS), sorted_scores, color=sorted_colors,
                  edgecolor="white", linewidth=0.5, alpha=0.85)

# shade pruned region
ax_bar.axhline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold:.1f}) — 50th percentile")
ax_bar.axhspan(ax_bar.get_ylim()[0], threshold, alpha=0.08, color="red")

# mark kept/pruned
for i, (bar, k) in enumerate(zip(bars, sorted_kept)):
    bar.set_edgecolor("#000" if k else "#aaa")
    bar.set_linewidth(1.5 if k else 0.5)
    bar.set_alpha(1.0 if k else 0.4)

ax_bar.set_xlabel("Graphs (sorted by anomaly score)", fontsize=10)
ax_bar.set_ylabel("Anomaly score\n(dist² from hypersphere center)", fontsize=10)
ax_bar.set_title("Panel B — Anomaly scores output by GIN Deep SVDD\n"
                 "Bright bars = KEPT (high score = informative)  |  Faded bars = PRUNED",
                 fontsize=10)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4CAF50", label="cluster"),
    Patch(facecolor="#2196F3", label="star"),
    Patch(facecolor="#FF9800", label="chain"),
    Patch(facecolor="#F44336", label="outlier"),
    Patch(facecolor="white",   edgecolor="black", linewidth=1.5, label="kept"),
    Patch(facecolor="gray",    alpha=0.4, label="pruned"),
]
ax_bar.legend(handles=legend_elements, loc="upper left", fontsize=8, ncol=3)

# ── Panel C: kept vs pruned side-by-side graphs ─────────────
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs[2], wspace=0.1)

show_kept   = kept_idx[:4]
show_pruned = pruned_idx[:4]

for col, idx in enumerate(show_kept):
    ax = fig.add_subplot(gs_bot[col])
    G = to_networkx(data_list[idx], to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.spring_layout(G, seed=0)
    nx.draw_networkx(G, pos, ax=ax, with_labels=False,
                     node_color="#2196F3", node_size=80,
                     edge_color="#333", width=1.0, alpha=0.9)
    ax.set_title(f"KEPT\n{types[idx]}\n{anomaly_scores[idx]:.1f}", fontsize=7, color="#1565C0")
    ax.axis("off")
    ax.patch.set_facecolor("#E3F2FD")

for col, idx in enumerate(show_pruned):
    ax = fig.add_subplot(gs_bot[col + 4])
    G = to_networkx(data_list[idx], to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.spring_layout(G, seed=0)
    nx.draw_networkx(G, pos, ax=ax, with_labels=False,
                     node_color="#EF9A9A", node_size=80,
                     edge_color="#888", width=0.8, alpha=0.6)
    ax.set_title(f"PRUNED\n{types[idx]}\n{anomaly_scores[idx]:.1f}", fontsize=7, color="#B71C1C")
    ax.axis("off")
    ax.patch.set_facecolor("#FFEBEE")

fig.text(0.5, 0.02,
         "Panel C — 4 kept graphs (blue) vs 4 pruned graphs (red)\n"
         "Kept = high anomaly score = harder/more informative samples  |  "
         "Pruned = near hypersphere center = typical/redundant",
         ha="center", fontsize=9, style="italic")

out_path = os.path.join(os.path.dirname(__file__), "screenshots", "pruning_visualization.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out_path}")
plt.show()