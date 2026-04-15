"""
Loads REAL data for each dataset (5% sample) and visualises graph structure.

Datasets loaded from actual files on disk:
  Cora      -> OFA/data/single_graph/Cora/cora.pt
  Pubmed    -> OFA/data/single_graph/Pubmed/pubmed.pt
  arxiv     -> OFA/cache_data/arxiv/ST/ogbn_arxiv/  (ogb format)
  WikiCS    -> OFA/data/single_graph/wikics/  (torch_geometric)
  FB15K237  -> OFA/data/KG/FB15K237/train.txt
  WN18RR    -> OFA/data/KG/WN18RR/train.txt
  chemmol   -> OFA/cache_data/dataset/  (HuggingFace cache)

Run:  C:/Users/salma/anaconda3/python.exe view_real_graphs.py
"""

import os, sys, json, random
import torch

# patch weights_only so ogb/pyg files load correctly on PyTorch >= 2.6
_orig_load = torch.load
def _load(*a, **kw): kw.setdefault("weights_only", False); return _orig_load(*a, **kw)
torch.load = _load

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

random.seed(42); np.random.seed(42); torch.manual_seed(42)

ROOT   = os.path.dirname(__file__)
OFA    = os.path.join(ROOT, "OFA")
sys.path.insert(0, OFA)

SAMPLE = 0.05   # 5% of data


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def induced_subgraph(edge_index, num_nodes, keep_nodes):
    """Return edges where BOTH endpoints are in keep_nodes set."""
    keep = set(keep_nodes.tolist())
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    edges = [(s, d) for s, d in zip(src, dst) if s in keep and d in keep]
    return edges

def sample_nodes(total, frac=SAMPLE):
    n = max(50, int(total * frac))
    return torch.randperm(total)[:n]

def build_nx(edges, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def draw_graph(ax, G, node_color, edge_color, title, info_lines,
               node_shape="o", directed=False, node_size=60):
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no edges\nin sample", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.2)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                               node_size=node_size, node_shape=node_shape, alpha=0.85)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color,
                               width=0.8, alpha=0.6,
                               arrows=directed, arrowsize=8)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.axis("off")
    info = "\n".join(info_lines)
    ax.text(0.01, 0.01, info, transform=ax.transAxes,
            fontsize=6.5, va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))


# ──────────────────────────────────────────────────────────────────────────────
# Loaders — each returns (G, stats_dict, node_feat_example, edge_feat_example)
# ──────────────────────────────────────────────────────────────────────────────

def load_cora():
    path = os.path.join(OFA, "data/single_graph/Cora/cora.pt")
    data = torch.load(path, weights_only=False)
    keep = sample_nodes(data.num_nodes)
    edges = induced_subgraph(data.edge_index, data.num_nodes, keep)
    G = build_nx(edges, directed=False)
    labels = [str(l) for l in data.label_names]
    return G, {
        "total nodes": data.num_nodes,
        "total edges": data.edge_index.shape[1] // 2,
        "sample nodes": len(keep),
        "sample edges": G.number_of_edges(),
        "classes (7)": ", ".join(labels[:3]) + "...",
        "task": "node classification",
    }, \
    "feature node. paper title and abstract:\n  " + str(data.raw_texts[0])[:80] + "...", \
    "feature edge. connected papers are\n  cited together by other papers."

def load_pubmed():
    path = os.path.join(OFA, "data/single_graph/Pubmed/pubmed.pt")
    data = torch.load(path, weights_only=False)
    keep = sample_nodes(data.num_nodes)
    edges = induced_subgraph(data.edge_index, data.num_nodes, keep)
    G = build_nx(edges, directed=False)
    with open(os.path.join(OFA, "data/single_graph/Pubmed/categories.csv")) as f:
        cats = [l.strip()[:60] for l in f.read().split("\n") if l.strip()]
    return G, {
        "total nodes": data.num_nodes,
        "total edges": data.edge_index.shape[1] // 2,
        "sample nodes": len(keep),
        "sample edges": G.number_of_edges(),
        "classes (3)": str(cats[:3]),
        "task": "node classification",
    }, \
    "feature node. paper title and abstract:\n  " + str(data.raw_texts[0])[:80] + "...", \
    "feature edge. connected papers are\n  cited together by other papers."

def load_arxiv():
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset("ogbn-arxiv",
                                root=os.path.join(OFA, "cache_data/arxiv/ST"))
    data = ds.data
    keep = sample_nodes(data.num_nodes)
    edges = induced_subgraph(data.edge_index, data.num_nodes, keep)
    G = build_nx(edges, directed=True)
    return G, {
        "total nodes": data.num_nodes,
        "total edges": data.edge_index.shape[1],
        "sample nodes": len(keep),
        "sample edges": G.number_of_edges(),
        "classes": "40 CS arXiv categories",
        "node feat": f"{data.x.shape[1]}-dim (Word2Vec avg)",
        "task": "node classification",
    }, \
    "feature node. paper title and abstract:\n  (768-dim Sentence-BERT in OFA pipeline)", \
    "feature edge. directed citation\n  (paper A cites paper B)"

def load_wikics():
    from torch_geometric.datasets import WikiCS
    wikics_path = os.path.join(OFA, "data/single_graph/wikics")
    ds = WikiCS(root=wikics_path)
    data = ds.data
    keep = sample_nodes(data.num_nodes)
    edges = induced_subgraph(data.edge_index, data.num_nodes, keep)
    G = build_nx(edges, directed=False)
    with open(os.path.join(wikics_path, "metadata.json")) as f:
        meta = json.load(f)
    label_names = list(meta["labels"].values())
    return G, {
        "total nodes": data.num_nodes,
        "total edges": data.edge_index.shape[1] // 2,
        "sample nodes": len(keep),
        "sample edges": G.number_of_edges(),
        "classes (10)": ", ".join(label_names[:3]) + "...",
        "task": "node classification",
    }, \
    "feature node. wikipedia entry name + content:\n  " + meta["nodes"][0]["title"][:60], \
    "feature edge. wikipedia hyperlink\n  between two pages."

def load_fb15k237():
    path = os.path.join(OFA, "data/KG/FB15K237/train.txt")
    triplets = []
    with open(path) as f:
        for line in f:
            h, r, t = line.strip().split()
            triplets.append((h, r, t))
    n = max(200, int(len(triplets) * SAMPLE))
    sample = random.sample(triplets, n)
    G = nx.DiGraph()
    for h, r, t in sample:
        G.add_edge(h, t, relation=r)
    G.remove_edges_from(nx.selfloop_edges(G))
    relations = list({r for _, r, _ in sample})
    return G, {
        "total triplets": len(triplets),
        "sample triplets": n,
        "sample nodes": G.number_of_nodes(),
        "sample edges": G.number_of_edges(),
        "relation types": "237 Freebase relations",
        "task": "link/relation prediction",
    }, \
    "feature node. entity description:\n  (Freebase entity + Wikidata label)", \
    f"feature edge. typed relation:\n  e.g. {relations[0]}"

def load_wn18rr():
    path = os.path.join(OFA, "data/KG/WN18RR/train.txt")
    triplets = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triplets.append(tuple(parts))
    n = max(200, int(len(triplets) * SAMPLE))
    sample = random.sample(triplets, n)
    G = nx.DiGraph()
    for h, r, t in sample:
        G.add_edge(h, t, relation=r)
    G.remove_edges_from(nx.selfloop_edges(G))
    # read entity names
    entity2text = {}
    epath = os.path.join(OFA, "data/KG/WN18RR/entity2text.txt")
    with open(epath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity2text[parts[0]] = parts[1]
    sample_rel = list({r for _, r, _ in sample})
    return G, {
        "total triplets": len(triplets),
        "sample triplets": n,
        "sample nodes": G.number_of_nodes(),
        "sample edges": G.number_of_edges(),
        "relation types": "11 WordNet relations",
        "example relation": sample_rel[0] if sample_rel else "N/A",
        "task": "link/relation prediction",
    }, \
    "feature node. WordNet synset:\n  e.g. '__hyponym.n.01': a related concept", \
    f"feature edge. lexical relation:\n  e.g. hypernym, meronym, similar_to..."

def load_chemmol():
    """Load real SMILES molecules directly from parquet (avoids HF path issues on Windows)."""
    import pandas as pd
    snap = os.path.normpath(os.path.join(OFA, "cache_data", "dataset",
                            "molecule_property_instruction", "data"))
    pq = os.path.normpath(os.path.join(snap, "chembl_pretraining-00000-of-00025-6a6e3f179bcd16a5.parquet"))
    df = pd.read_parquet(pq)
    smiles_col = "graph"   # column name used in gen_data.py
    class FakeDS:
        def __init__(self, df): self._df = df
        def __len__(self): return len(self._df)
        def __getitem__(self, i): return {smiles_col: self._df.iloc[i][smiles_col]}
    ds = FakeDS(df)
    total = len(ds)
    total = len(ds)
    n = max(20, int(total * SAMPLE))
    indices = random.sample(range(total), n)

    from rdkit import Chem
    import sys as _sys
    # suppress rdkit warnings
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    graphs = []
    smiles_example = None
    for i in indices:
        smi = ds[i]["graph"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if smiles_example is None:
            smiles_example = smi
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                       bond_type=str(bond.GetBondTypeAsDouble()))
        graphs.append(G)

    # pick one representative molecule to draw
    # pick medium-size for clarity
    graphs.sort(key=lambda g: g.number_of_nodes())
    mid = graphs[len(graphs) // 2] if graphs else nx.Graph()

    return mid, {
        "total molecules": total,
        "sample molecules": n,
        "example atoms": mid.number_of_nodes(),
        "example bonds": mid.number_of_edges(),
        "node features": "9-dim: atom, chirality, degree, charge...",
        "edge features": "3-dim: bond type, stereo, conjugated",
        "task": "molecular property prediction",
    }, \
    f"feature node. atom:\n  e.g. 'Carbon, sp2, aromatic, in ring'", \
    f"feature edge. chemical bond:\n  e.g. 'SINGLE/DOUBLE/AROMATIC bond'"


# ──────────────────────────────────────────────────────────────────────────────
# Load all datasets
# ──────────────────────────────────────────────────────────────────────────────

print("Loading real datasets (5% sample each)...")
loaders = [
    ("Cora",     load_cora,     "#4CAF50", "o", False),
    ("Pubmed",   load_pubmed,   "#81C784", "o", False),
    ("arxiv",    load_arxiv,    "#AED581", "o", True),
    ("WikiCS",   load_wikics,   "#DCE775", "o", False),
    ("FB15K237", load_fb15k237, "#FF8A65", "s", True),
    ("WN18RR",   load_wn18rr,   "#E57373", "s", True),
    ("chemmol",  load_chemmol,  "#7986CB", "^", False),
]

results = {}
for name, loader, color, shape, directed in loaders:
    print(f"  Loading {name}...", end=" ", flush=True)
    try:
        G, stats, node_feat, edge_feat = loader()
        results[name] = (G, stats, node_feat, edge_feat, color, shape, directed)
        print(f"OK — {stats.get('sample nodes', G.number_of_nodes())} nodes, "
              f"{stats.get('sample edges', G.number_of_edges())} edges")
    except Exception as e:
        print(f"FAILED: {e}")
        results[name] = None


# ──────────────────────────────────────────────────────────────────────────────
# Figure: 3 rows x 7 cols
#   Row 1 — graph drawing
#   Row 2 — dataset stats table
#   Row 3 — node & edge feature examples
# ──────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(28, 18))
fig.suptitle(
    "Real Graph Data — 5% Sample per Dataset\n"
    "Node = paper / entity / atom   |   Edge = citation / relation / bond",
    fontsize=14, fontweight="bold"
)

gs = gridspec.GridSpec(3, 7, figure=fig, hspace=0.55, wspace=0.25)

for col, (name, loader, color, shape, directed) in enumerate(loaders):
    entry = results.get(name)

    # ── Row 0: graph drawing ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, col])
    if entry is None:
        ax.text(0.5, 0.5, f"{name}\nload failed", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="red")
        ax.axis("off")
        continue

    G, stats, node_feat, edge_feat, color, shape, directed = entry
    g_draw = G
    if G.number_of_nodes() > 120:          # cap for readability
        keep = list(G.nodes())[:120]
        g_draw = G.subgraph(keep)

    ec = "#E53935" if shape == "s" else ("#5C6BC0" if shape == "^" else "#444")
    draw_graph(ax, g_draw, color, ec,
               title=name,
               info_lines=[],
               node_shape=shape, directed=directed,
               node_size=40 if G.number_of_nodes() > 60 else 80)

    # dataset type label
    dtype = ("KG" if shape == "s" else ("Molecule" if shape == "^" else "Citation"))
    ax.set_xlabel(dtype, fontsize=8, labelpad=2)

    # ── Row 1: stats table ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, col])
    ax2.axis("off")
    rows = list(stats.items())
    table_data = [[k, str(v)] for k, v in rows]
    t = ax2.table(cellText=table_data,
                  colLabels=["property", "value"],
                  cellLoc="left", loc="center",
                  bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(6.5)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor(color)
            cell.set_text_props(fontweight="bold", fontsize=7)
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
    ax2.set_title(f"{name} — stats", fontsize=7, fontweight="bold")

    # ── Row 2: feature description ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, col])
    ax3.axis("off")
    ax3.set_facecolor("#FAFAFA")
    ax3.text(0.5, 0.75, "NODE FEATURE", ha="center", fontsize=7,
             fontweight="bold", transform=ax3.transAxes, color="#1565C0")
    ax3.text(0.5, 0.55, node_feat, ha="center", fontsize=6.2,
             transform=ax3.transAxes, wrap=True, va="top",
             bbox=dict(boxstyle="round", fc="#E3F2FD", alpha=0.8))
    ax3.text(0.5, 0.28, "EDGE FEATURE", ha="center", fontsize=7,
             fontweight="bold", transform=ax3.transAxes, color="#B71C1C")
    ax3.text(0.5, 0.08, edge_feat, ha="center", fontsize=6.2,
             transform=ax3.transAxes, wrap=True, va="top",
             bbox=dict(boxstyle="round", fc="#FFEBEE", alpha=0.8))

fig.text(0.5, 0.005,
         "Row 1: Real 5% subgraph (circle=paper, square=KG entity, triangle=atom)  |  "
         "Row 2: Dataset statistics  |  "
         "Row 3: What the node & edge features represent in the OFA pipeline",
         ha="center", fontsize=9, style="italic")

out = os.path.join(ROOT, "screenshots", "real_graphs.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.show()