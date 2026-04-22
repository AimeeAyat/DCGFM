"""
Synthetic visualization of all 6 dataset types used in DCGFM:
  arxiv, cora, pubmed, wikics  → citation/social graphs   (node classification)
  FB15K237, WN18RR             → knowledge graphs          (link/relation prediction)
  chemmol                      → molecule graphs           (graph classification)

Each dataset's subgraphs differ in:
  - node semantics   (papers / entities / atoms)
  - edge semantics   (citations / relations / bonds)
  - graph structure  (sparse citation vs dense KG vs ring/chain molecule)
  - feature dim      (768-dim SBERT for text / atomic features for mol)

Run:  C:/Users/salma/anaconda3/python.exe visualize_datasets.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import random

random.seed(0); np.random.seed(0); torch.manual_seed(0)

FEAT_DIM = 32   # 768 in real pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Dataset simulators — each mimics what the real gen_data.py produces
# ──────────────────────────────────────────────────────────────────────────────

def sim_citation(n=12, density=0.25):
    """
    arxiv / cora / pubmed / wikics
    ─────────────────────────────
    One BIG graph (all papers) — training extracts 2-hop subgraphs around each node.
    Nodes = papers  |  Edges = citations
    Node features = 768-dim Sentence-BERT("title + abstract")
    Labels = paper category (arxiv: 40 CS classes, cora: 7, pubmed: 3, wikics: 10)
    """
    G = nx.erdos_renyi_graph(n, density, seed=0, directed=True)
    G = nx.DiGraph(G)
    edges = list(G.edges())
    if not edges:
        edges = [(0, 1)]
    ei = torch.tensor(edges, dtype=torch.long).t()
    x = torch.randn(n, FEAT_DIM)          # SBERT embeddings of title+abstract
    y = torch.randint(0, 7, (n,))         # paper category label
    return Data(x=x, edge_index=ei, y=y, num_nodes=n), G

def sim_kg(n=10, n_relations=4):
    """
    FB15K237 / WN18RR
    ─────────────────
    One BIG knowledge graph — training extracts 2-hop subgraphs around each triplet.
    Nodes = entities (people/places/concepts/WordNet synsets)
    Edges = typed relations  (e.g. 'bornIn', 'partOf', 'hypernymOf')
    Node features = 768-dim SBERT("entity name + description")
    Edge features = relation type embedding
    Task = predict relation type between two entities (link prediction)
    """
    G = nx.gnm_random_graph(n, n * 2, seed=1)
    edges = list(G.edges())
    if not edges:
        edges = [(0, 1)]
    ei = torch.tensor(edges, dtype=torch.long).t()
    x = torch.randn(n, FEAT_DIM)                              # entity embeddings
    edge_types = torch.randint(0, n_relations, (len(edges),)) # relation type per edge
    return Data(x=x, edge_index=ei, edge_attr=edge_types, num_nodes=n), G

def sim_molecule(n_atoms=12):
    """
    chemmol  (chemblpre / chempcba / chemhiv)
    ─────────────────────────────────────────
    Each molecule IS its own graph (not a subgraph of a big graph).
    Nodes = atoms       (C, N, O, F, ...)
    Edges = bonds       (single, double, triple, aromatic)
    Node features = atomic number, chirality, degree, formal charge, ... (9-dim OGB)
    Edge features = bond type, stereo config (3-dim OGB)
    Task = molecular property prediction (toxicity, activity, HIV inhibition)

    Real structure: ring + chain (typical organic molecule topology)
    """
    ring_size = min(6, n_atoms)
    G = nx.cycle_graph(ring_size)             # benzene-like ring
    for i in range(ring_size, n_atoms):       # attach chain substituents
        G.add_edge(i % ring_size, i)
    edges = list(G.edges())
    ei = torch.tensor(edges, dtype=torch.long).t()
    ei = torch.cat([ei, ei.flip(0)], dim=1)  # undirected bonds
    x = torch.randint(1, 9, (n_atoms, 9)).float()   # atomic features (OGB format)
    edge_attr = torch.randint(0, 4, (ei.shape[1],)) # bond type
    y = torch.randint(0, 2, (1, 10)).float()         # multi-label property
    return Data(x=x, edge_index=ei, edge_attr=edge_attr, y=y, num_nodes=n_atoms), G


# ──────────────────────────────────────────────────────────────────────────────
# Build datasets (N subgraphs each)
# ──────────────────────────────────────────────────────────────────────────────

N = 20
datasets = {
    "arxiv\n(citation, 40 classes\nSBERT node feat)":         [sim_citation(random.randint(8,16), 0.2) for _ in range(N)],
    "cora\n(citation, 7 classes\nSBERT node feat)":            [sim_citation(random.randint(6,12), 0.3) for _ in range(N)],
    "pubmed\n(citation, 3 classes\nSBERT node feat)":          [sim_citation(random.randint(8,14), 0.15) for _ in range(N)],
    "wikics\n(social/citation, 10 classes\nSBERT node feat)":  [sim_citation(random.randint(8,18), 0.18) for _ in range(N)],
    "FB15K237\n(KG, typed edges\nentity descriptions)":        [sim_kg(random.randint(6,14), 237) for _ in range(N)],
    "WN18RR\n(KG, WordNet\nhypernym/meronym)":                 [sim_kg(random.randint(5,12), 11) for _ in range(N)],
    "chemmol\n(molecules\natomic features)":                   [sim_molecule(random.randint(8,16)) for _ in range(N)],
}

palette = ["#4CAF50","#81C784","#AED581","#DCE775","#FF8A65","#E57373","#7986CB"]


# ──────────────────────────────────────────────────────────────────────────────
# Run hard pruning on each dataset
# ──────────────────────────────────────────────────────────────────────────────

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import InMemoryDataset, DataLoader
import pytorch_lightning as pl

class _GIN(nn.Module):
    def __init__(self, nfeat, nhid=64, nlayer=3, act=ReLU()):
        super().__init__()
        self.transform = Sequential(Linear(nfeat, nhid), BatchNorm1d(nhid))
        self.pooling = global_mean_pool
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(nlayer):
            mlp = Sequential(Linear(nhid, nhid), act, Linear(nhid, nhid))
            self.convs.append(GINConv(mlp))
            self.bns.append(BatchNorm1d(nhid))
        self.act = act

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = self.transform(x)
        embeds = [self.pooling(x, batch)]
        for conv, bn in zip(self.convs, self.bns):
            x = self.act(bn(conv(x, ei)))
            embeds.append(self.pooling(x, batch))
        return torch.stack(embeds)

class _SVDD(pl.LightningModule):
    def __init__(self, nfeat, nhid=64, nlayer=3):
        super().__init__()
        self.gin = _GIN(nfeat, nhid, nlayer)
        self.register_buffer('center', torch.zeros(nhid))
        self.radius = 0; self.nu = 1
        self._ep0, self._test = [], []

    def forward(self, data):
        e = self.gin(data).sum(0)
        return torch.sum((e - self.center)**2, dim=1) - self.radius**2

    def training_step(self, batch, _):
        if self.current_epoch == 0:
            e = self.gin(batch)
            loss = torch.zeros(1, requires_grad=True, device=self.device)
            self._ep0.append(e.detach()); return loss
        s = self(batch)
        loss = self.radius**2 + (1/self.nu)*F.relu(s).mean()
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch == 0 and self._ep0:
            self.center = torch.cat(self._ep0, 1).mean(1).sum(0)
        self._ep0 = []

    def test_step(self, batch, _):
        self._test.append(self(batch))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class _DS(InMemoryDataset):
    def __init__(self, lst):
        super().__init__('.')
        self.data, self.slices = self.collate(lst)
    raw_file_names = processed_file_names = []
    def download(self): pass
    def process(self): pass

def run_pruning(data_list, epochs=12):
    ds = _DS(data_list)
    tdl = DataLoader(ds, batch_size=8, shuffle=True)
    vdl = DataLoader(ds, batch_size=8, shuffle=False)
    model = _SVDD(ds[0].x.shape[1])
    trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=epochs,
                         logger=False, enable_progress_bar=False)
    trainer.fit(model, tdl)
    trainer.test(model, vdl)
    return torch.cat(model._test).cpu().detach().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

print("Running hard pruning for all 7 datasets...")
all_scores = {}
for name, pairs in datasets.items():
    dlist = [p[0] for p in pairs]
    scores = run_pruning(dlist)
    all_scores[name] = scores
    short = name.split('\n')[0]
    print(f"  {short:12s}  min={scores.min():.1f}  max={scores.max():.1f}  mean={scores.mean():.1f}")

fig = plt.figure(figsize=(24, 18))
fig.suptitle("DCGFM Hard Pruning — All 7 Datasets\n"
             "Each dataset produces different graph structures → different anomaly score distributions",
             fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5)

# ── Row 1: example graph per dataset ─────────────────────────────────────────
gs_top = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[0], wspace=0.15)
for col, ((name, pairs), color) in enumerate(zip(datasets.items(), palette)):
    ax = fig.add_subplot(gs_top[col])
    _, G = pairs[0]
    G_u = G.to_undirected() if G.is_directed() else G
    G_u.remove_edges_from(nx.selfloop_edges(G_u))
    pos = nx.spring_layout(G_u, seed=42)

    is_kg  = "FB15K" in name or "WN18" in name
    is_mol = "chem" in name

    edge_color = "#E53935" if is_kg else ("#795548" if is_mol else "#555")
    node_shape = "s" if is_kg else ("^" if is_mol else "o")
    nx.draw_networkx_nodes(G_u, pos, ax=ax, node_color=color,
                           node_size=160, node_shape=node_shape, alpha=0.9)
    nx.draw_networkx_edges(G_u, pos, ax=ax, edge_color=edge_color,
                           width=1.5, alpha=0.7, arrows=G.is_directed(),
                           arrowsize=10)

    tag = "citation" if not is_kg and not is_mol else ("KG triplets" if is_kg else "molecule")
    ax.set_title(name + f"\n[{tag}]", fontsize=7.5)
    ax.axis("off")

fig.text(0.5, 0.685,
         "Row 1 — One example subgraph per dataset  "
         "(circle=citation node, square=KG entity, triangle=atom)",
         ha="center", fontsize=9, style="italic")

# ── Row 2: score distributions (violin) ──────────────────────────────────────
ax_v = fig.add_subplot(gs[1])
score_list  = [all_scores[n] for n in all_scores]
short_names = [n.split('\n')[0] for n in all_scores]

parts = ax_v.violinplot(score_list, positions=range(len(score_list)),
                         showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(palette[i]); pc.set_alpha(0.7)
parts['cmeans'].set_color('black')
parts['cmedians'].set_color('red')

ax_v.set_xticks(range(len(short_names)))
ax_v.set_xticklabels(short_names, fontsize=9)
ax_v.set_ylabel("Anomaly score (dist² from center)", fontsize=10)
ax_v.set_title("Row 2 — Anomaly score distribution per dataset\n"
               "Different datasets → different score ranges (KG/mol spread wider due to structural diversity)",
               fontsize=10)
ax_v.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# ── Row 3: kept vs pruned bar per dataset ─────────────────────────────────────
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[2], wspace=0.3)
for col, (name, color) in enumerate(zip(all_scores, palette)):
    ax = fig.add_subplot(gs_bot[col])
    scores = all_scores[name]
    thresh = np.percentile(scores, 50)
    kept   = scores >= thresh
    idx    = np.argsort(scores)
    colors = [color if kept[i] else "#BDBDBD" for i in idx]
    ax.bar(range(N), scores[idx], color=colors, edgecolor='white', linewidth=0.4)
    ax.axhline(thresh, color='black', linestyle='--', linewidth=1)
    short = name.split('\n')[0]
    ax.set_title(f"{short}\nkept={kept.sum()} pruned={N-kept.sum()}", fontsize=8)
    ax.set_xlabel("graphs (sorted)", fontsize=7)
    ax.tick_params(labelsize=7)
    if col == 0:
        ax.set_ylabel("anomaly score", fontsize=8)

fig.text(0.5, 0.01,
         "Row 3 — Per-dataset kept (colored) vs pruned (gray) bars  |  "
         "Dashed line = 50th percentile threshold  |  "
         "Hard samples (high score) are kept for training",
         ha="center", fontsize=9, style="italic")

import os
out = os.path.join(os.path.dirname(__file__), "screenshots", "all_datasets_pruning.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.show()