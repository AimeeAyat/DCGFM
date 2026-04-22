"""
Hard Pruning Demo — ChemMol Dataset
====================================
This script demonstrates EXACTLY how hard pruning works on real molecule graphs:

  Step 1: Load SMILES strings from ChEMBL parquet
  Step 2: Convert each SMILES → PyG Data graph (nodes=atoms, edges=bonds)
  Step 3: Train Deep SVDD (GIN encoder learns a hypersphere)
  Step 4: Score every graph — distance from hypersphere center
  Step 5: Threshold: keep graphs with score > median (anomalous = informative)
  Step 6: Log and visualize retained vs pruned molecules

Run with:
    C:/Users/salma/anaconda3/python.exe hard_prune_demo.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
OFA  = os.path.join(ROOT, "OFA")
sys.path.insert(0, OFA)
sys.path.insert(0, os.path.join(OFA, "data/chemmol"))

# ── monkey-patch torch.load for PyTorch >= 2.6 ────────────────────────────────
import torch
_orig_load = torch.load
def _safe_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _safe_load

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

# ── output folder ─────────────────────────────────────────────────────────────
OUT = os.path.join(ROOT, "visualizations", "output")
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("HARD PRUNING DEMO — ChemMol (ChEMBL molecules)")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# STEP 1: Load SMILES from parquet
# ════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading SMILES from ChEMBL parquet...")

PQ_PATH = os.path.normpath(os.path.join(
    OFA, "cache_data", "dataset",
    "molecule_property_instruction", "data",
    "chembl_pretraining-00000-of-00025-6a6e3f179bcd16a5.parquet"
))
df = pd.read_parquet(PQ_PATH)
all_smiles = df["graph"].dropna().tolist()

# use first 200 molecules for a fast demo
N = 200
smiles_subset = all_smiles[:N]
print(f"  Total in shard: {len(all_smiles)} | Using first {N} for demo")

# ════════════════════════════════════════════════════════════════
# STEP 2: Convert SMILES -> PyG Data graphs
# ════════════════════════════════════════════════════════════════
print("\n[STEP 2] Converting SMILES -> molecular graphs...")
print("  Each atom becomes a node with 9 numeric features:")
print("  [atomic_num, degree, formal_charge, n_Hs, radical_e, hybridization,")
print("   is_aromatic, is_in_ring, chirality]")

def smiles_to_pyg(smiles):
    """Convert SMILES string to a PyG Data object with numeric atom features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # numeric atom features (9-dim) — same categories OGB uses
    hybridization_map = {
        Chem.rdchem.HybridizationType.SP:    0,
        Chem.rdchem.HybridizationType.SP2:   1,
        Chem.rdchem.HybridizationType.SP3:   2,
        Chem.rdchem.HybridizationType.SP3D:  3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    chirality_map = {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED:      0,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:   1,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:  2,
        Chem.rdchem.ChiralType.CHI_OTHER:            3,
    }

    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            hybridization_map.get(atom.GetHybridization(), 5),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            chirality_map.get(atom.GetChiralTag(), 0),
        ])

    if len(node_feats) == 0:
        return None

    x = torch.tensor(node_feats, dtype=torch.float)

    edges_src, edges_dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges_src += [i, j]
        edges_dst += [j, i]

    if len(edges_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        num_nodes=x.shape[0],
        smiles=smiles,
    )

data_list = []
failed   = 0
for i, smi in enumerate(smiles_subset):
    d = smiles_to_pyg(smi)
    if d is not None and d.x.shape[0] > 1:  # need at least 2 atoms
        data_list.append(d)
    else:
        failed += 1

print(f"  Converted: {len(data_list)} valid graphs | Failed/skipped: {failed}")
print(f"  Feature dim: {data_list[0].x.shape[1]}")
print(f"  Atom counts: min={min(d.num_nodes for d in data_list)}, "
      f"max={max(d.num_nodes for d in data_list)}, "
      f"mean={np.mean([d.num_nodes for d in data_list]):.1f}")

# ════════════════════════════════════════════════════════════════
# STEP 3: Build GIN model (Deep SVDD encoder)
# ════════════════════════════════════════════════════════════════
print("\n[STEP 3] Building GIN (Graph Isomorphism Network) encoder...")
print("  Architecture:")
print("    Input -> Linear(9->128) + BatchNorm")
print("    x3 GINConv layers: MLP(128->128->128) + ReLU + BatchNorm")
print("    global_mean_pool -> 128-dim graph embedding per layer")
print("    sum all 4 layer embeddings -> 128-dim final embedding")
print("  Loss: Deep SVDD radius loss (push graphs inside a hypersphere)")

NFEAT  = data_list[0].x.shape[1]   # 9
NHID   = 128
NLAYER = 3

class GIN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, nlayer):
        super().__init__()
        self.transform = Sequential(Linear(nfeat, nhid), BatchNorm1d(nhid))
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = ReLU()
        for _ in range(nlayer):
            nn_mlp = Sequential(Linear(nhid, nhid), self.act, Linear(nhid, nhid))
            self.convs.append(GINConv(nn_mlp))
            self.bns.append(BatchNorm1d(nhid))
        self.pool = global_mean_pool

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = self.transform(x)
        layer_embeds = [self.pool(x, batch)]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = self.act(x)
            x = bn(x)
            layer_embeds.append(self.pool(x, batch))
        # sum all layer embeddings (same as 'sum' mode in hard_prune_module.py)
        return torch.stack(layer_embeds).sum(dim=0)   # [N_graphs, nhid]

model = GIN_Encoder(NFEAT, NHID, NLAYER)
device = torch.device("cpu")
model = model.to(device)
print(f"  Running on: {device}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ════════════════════════════════════════════════════════════════
# STEP 4: Deep SVDD training (learn hypersphere center)
# ════════════════════════════════════════════════════════════════
print("\n[STEP 4] Training Deep SVDD...")
print("  Epoch 0: forward pass only — compute hypersphere center")
print("  Epoch 1+: minimize radius² + (1/nu) * mean(relu(dist² - radius²))")

class SimpleDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)
    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return []
    def download(self): pass
    def process(self): pass

dataset = SimpleDataset(data_list)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
center = None
radius = torch.tensor(0.0)
nu = 1.0
MAX_EPOCHS = 15

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0.0
    all_embeds = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        embeds = model(batch)

        if epoch == 0:
            # epoch 0: just collect embeddings to compute center
            all_embeds.append(embeds.detach())
            loss = torch.zeros(1, requires_grad=True)
        else:
            dist   = torch.sum((embeds - center) ** 2, dim=1)
            scores = dist - radius ** 2
            loss   = radius ** 2 + (1 / nu) * torch.mean(F.relu(scores))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch == 0:
        # set center = mean of all embeddings from epoch 0
        center = torch.cat(all_embeds, dim=0).mean(dim=0)
        print(f"  Epoch 0: center initialized (norm={center.norm():.4f})")
    else:
        print(f"  Epoch {epoch:2d}: loss={epoch_loss/len(loader):.6f}  "
              f"radius={radius.item():.4f}")

# ════════════════════════════════════════════════════════════════
# STEP 5: Score every graph
# ════════════════════════════════════════════════════════════════
print("\n[STEP 5] Scoring all graphs (anomaly = dist² from center - radius²)...")
print("  HIGH score = far from center = anomalous = informative -> KEEP")
print("  LOW score  = close to center = normal     = redundant  -> PRUNE")

model.eval()
test_loader  = DataLoader(dataset, batch_size=32, shuffle=False)
all_scores   = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        embeds = model(batch)
        dist   = torch.sum((embeds - center) ** 2, dim=1)
        scores = (dist - radius ** 2).cpu().numpy()
        all_scores.extend(scores)

all_scores = np.array(all_scores)

print(f"\n  Score statistics over {len(all_scores)} graphs:")
print(f"    min    = {all_scores.min():.4f}")
print(f"    max    = {all_scores.max():.4f}")
print(f"    mean   = {all_scores.mean():.4f}")
print(f"    median = {np.median(all_scores):.4f}")
print(f"    std    = {all_scores.std():.4f}")

# ════════════════════════════════════════════════════════════════
# STEP 6: Prune — keep top 50% by score
# ════════════════════════════════════════════════════════════════
KEEP_RATIO = 0.5
threshold  = np.percentile(all_scores, (1 - KEEP_RATIO) * 100)

retained_idx = np.where(all_scores >= threshold)[0]
pruned_idx   = np.where(all_scores <  threshold)[0]

print(f"\n[STEP 6] Pruning (keep top {int(KEEP_RATIO*100)}% by anomaly score)...")
print(f"  Threshold  = {threshold:.4f}")
print(f"  RETAINED   = {len(retained_idx)} graphs ({len(retained_idx)/len(data_list)*100:.1f}%)")
print(f"  PRUNED     = {len(pruned_idx)} graphs ({len(pruned_idx)/len(data_list)*100:.1f}%)")

# ── Detailed per-graph logging ────────────────────────────────────────────────
print("\n  Top 10 RETAINED (most informative):")
top_retained = retained_idx[np.argsort(all_scores[retained_idx])[::-1]][:10]
for rank, idx in enumerate(top_retained, 1):
    mol = Chem.MolFromSmiles(data_list[idx].smiles)
    n_atoms = mol.GetNumAtoms() if mol else "?"
    n_rings = rdMolDescriptors.CalcNumRings(mol) if mol else "?"
    print(f"    #{rank:2d}  idx={idx:4d}  score={all_scores[idx]:+.4f}  "
          f"atoms={n_atoms}  rings={n_rings}  smiles={data_list[idx].smiles[:40]}")

print("\n  Top 10 PRUNED (most redundant):")
top_pruned = pruned_idx[np.argsort(all_scores[pruned_idx])][:10]
for rank, idx in enumerate(top_pruned, 1):
    mol = Chem.MolFromSmiles(data_list[idx].smiles)
    n_atoms = mol.GetNumAtoms() if mol else "?"
    n_rings = rdMolDescriptors.CalcNumRings(mol) if mol else "?"
    print(f"    #{rank:2d}  idx={idx:4d}  score={all_scores[idx]:+.4f}  "
          f"atoms={n_atoms}  rings={n_rings}  smiles={data_list[idx].smiles[:40]}")

# ── Feature comparison ────────────────────────────────────────────────────────
retained_sizes = [data_list[i].num_nodes for i in retained_idx]
pruned_sizes   = [data_list[i].num_nodes for i in pruned_idx]
retained_edges = [data_list[i].edge_index.shape[1]//2 for i in retained_idx]
pruned_edges   = [data_list[i].edge_index.shape[1]//2 for i in pruned_idx]

retained_rings = []
pruned_rings   = []
for i in retained_idx:
    mol = Chem.MolFromSmiles(data_list[i].smiles)
    if mol: retained_rings.append(rdMolDescriptors.CalcNumRings(mol))
for i in pruned_idx:
    mol = Chem.MolFromSmiles(data_list[i].smiles)
    if mol: pruned_rings.append(rdMolDescriptors.CalcNumRings(mol))

print("\n  Structural comparison — RETAINED vs PRUNED:")
print(f"    {'Metric':<20} {'RETAINED':>12} {'PRUNED':>12}")
print(f"    {'-'*44}")
print(f"    {'avg atoms':<20} {np.mean(retained_sizes):>12.2f} {np.mean(pruned_sizes):>12.2f}")
print(f"    {'avg bonds':<20} {np.mean(retained_edges):>12.2f} {np.mean(pruned_edges):>12.2f}")
print(f"    {'avg rings':<20} {np.mean(retained_rings):>12.2f} {np.mean(pruned_rings):>12.2f}")
print(f"    {'avg score':<20} {all_scores[retained_idx].mean():>12.4f} {all_scores[pruned_idx].mean():>12.4f}")

# ════════════════════════════════════════════════════════════════
# VISUALS
# ════════════════════════════════════════════════════════════════
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

def mol_to_img(smiles, size=250):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))

# ── Visual A: Anomaly score distribution ─────────────────────────────────────
print("\n[VISUAL A] Score distribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Deep SVDD Anomaly Scores — ChemMol\n"
             "High score = far from hypersphere center = informative = KEPT",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.hist(all_scores[pruned_idx],   bins=25, color="#EF5350", alpha=0.7,
        label=f"PRUNED ({len(pruned_idx)})", density=True)
ax.hist(all_scores[retained_idx], bins=25, color="#4CAF50", alpha=0.7,
        label=f"RETAINED ({len(retained_idx)})", density=True)
ax.axvline(threshold, color="navy", linestyle="--", linewidth=2,
           label=f"Threshold = {threshold:.3f}")
ax.set_xlabel("Anomaly Score (dist² from center − radius²)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Score distribution — retained vs pruned", fontsize=10)
ax.legend(fontsize=9)

ax = axes[1]
sorted_idx = np.argsort(all_scores)
colors = ["#4CAF50" if all_scores[i] >= threshold else "#EF5350"
          for i in sorted_idx]
ax.bar(range(len(sorted_idx)), all_scores[sorted_idx], color=colors, width=1.0)
ax.axhline(threshold, color="navy", linestyle="--", linewidth=1.5,
           label=f"Threshold = {threshold:.3f}")
green_p = mpatches.Patch(color="#4CAF50", label="RETAINED (informative)")
red_p   = mpatches.Patch(color="#EF5350", label="PRUNED (redundant)")
ax.legend(handles=[green_p, red_p], fontsize=9)
ax.set_xlabel("Graphs sorted by score", fontsize=9)
ax.set_ylabel("Anomaly Score", fontsize=9)
ax.set_title("Every graph ranked — green = keep, red = prune", fontsize=10)

fig.savefig(os.path.join(OUT, "hp_A_score_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> hp_A_score_distribution.png")

# ── Visual B: Top retained vs top pruned molecules ────────────────────────────
print("[VISUAL B] Top retained vs top pruned molecules...")
fig, axes = plt.subplots(2, 8, figsize=(24, 8))
fig.suptitle("Hard Pruning — TOP RETAINED (green) vs TOP PRUNED (red) Molecules\n"
             "Retained = highest anomaly score | Pruned = lowest anomaly score",
             fontsize=12, fontweight="bold")

top_r = retained_idx[np.argsort(all_scores[retained_idx])[::-1]][:8]
top_p = pruned_idx[np.argsort(all_scores[pruned_idx])][:8]

for col, idx in enumerate(top_r):
    ax = axes[0][col]
    img = mol_to_img(data_list[idx].smiles)
    if img:
        ax.imshow(img)
    mol = Chem.MolFromSmiles(data_list[idx].smiles)
    n_a = mol.GetNumAtoms() if mol else "?"
    n_r = rdMolDescriptors.CalcNumRings(mol) if mol else "?"
    ax.set_title(f"score={all_scores[idx]:+.3f}\natoms={n_a} rings={n_r}",
                 fontsize=7, color="#1B5E20")
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#4CAF50"); spine.set_linewidth(3)

for col, idx in enumerate(top_p):
    ax = axes[1][col]
    img = mol_to_img(data_list[idx].smiles)
    if img:
        ax.imshow(img)
    mol = Chem.MolFromSmiles(data_list[idx].smiles)
    n_a = mol.GetNumAtoms() if mol else "?"
    n_r = rdMolDescriptors.CalcNumRings(mol) if mol else "?"
    ax.set_title(f"score={all_scores[idx]:+.3f}\natoms={n_a} rings={n_r}",
                 fontsize=7, color="#B71C1C")
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#EF5350"); spine.set_linewidth(3)

axes[0][0].set_ylabel("RETAINED\n(informative)", fontsize=10,
                       color="#1B5E20", fontweight="bold")
axes[1][0].set_ylabel("PRUNED\n(redundant)", fontsize=10,
                       color="#B71C1C", fontweight="bold")

fig.savefig(os.path.join(OUT, "hp_B_retained_vs_pruned_molecules.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> hp_B_retained_vs_pruned_molecules.png")

# ── Visual C: Structural stats comparison ────────────────────────────────────
print("[VISUAL C] Structural stats comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Structural Differences — Retained vs Pruned Molecules",
             fontsize=12, fontweight="bold")

bins = 20
for ax, (r_data, p_data, xlabel, title) in zip(axes, [
    (retained_sizes, pruned_sizes, "# atoms",  "Atom count distribution"),
    (retained_edges, pruned_edges, "# bonds",  "Bond count distribution"),
    (retained_rings, pruned_rings, "# rings",  "Ring count distribution"),
]):
    ax.hist(p_data, bins=bins, color="#EF5350", alpha=0.65, label="PRUNED",   density=True)
    ax.hist(r_data, bins=bins, color="#4CAF50", alpha=0.65, label="RETAINED", density=True)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

fig.savefig(os.path.join(OUT, "hp_C_structural_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> hp_C_structural_comparison.png")

# ── Visual D: Graph topology — one retained vs one pruned ─────────────────────
print("[VISUAL D] Graph topology examples...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Graph Topology — One Retained vs One Pruned Molecule\n"
             "(nodes = atoms colored by element, edges = bonds)",
             fontsize=12, fontweight="bold")

color_map = {"C":"#64B5F6","O":"#EF5350","N":"#4CAF50",
             "S":"#FFEB3B","F":"#CE93D8","Cl":"#80DEEA","Br":"#FFCC80"}

def draw_mol_graph(ax, data_item, score, label, border_color):
    mol = Chem.MolFromSmiles(data_item.smiles)
    if mol is None: return
    G = nx.Graph()
    atom_colors = []
    atom_labels = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        G.add_node(idx)
        atom_labels[idx] = sym
        atom_colors.append(color_map.get(sym, "#B0BEC5"))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, ax=ax, labels=atom_labels,
                     node_color=atom_colors, node_size=400,
                     font_size=8, font_weight="bold", edge_color="#555", width=1.5)
    n_atoms = mol.GetNumAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    ax.set_title(f"{label}\nscore={score:+.4f}  atoms={n_atoms}  rings={n_rings}\n"
                 f"{data_item.smiles[:45]}", fontsize=9)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(border_color); spine.set_linewidth(3)

best_r = top_r[0]
worst_p = top_p[0]

draw_mol_graph(axes[0], data_list[best_r],  all_scores[best_r],
               "RETAINED (most informative)", "#4CAF50")
draw_mol_graph(axes[1], data_list[worst_p], all_scores[worst_p],
               "PRUNED (most redundant)", "#EF5350")

element_legend = [mpatches.Patch(color=c, label=el)
                  for el, c in color_map.items()]
fig.legend(handles=element_legend, loc="lower center", ncol=7,
           fontsize=8, title="Element color")

fig.savefig(os.path.join(OUT, "hp_D_graph_topology.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> hp_D_graph_topology.png")

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total molecules processed : {len(data_list)}")
print(f"  RETAINED (informative)    : {len(retained_idx)} ({len(retained_idx)/len(data_list)*100:.0f}%)")
print(f"  PRUNED   (redundant)      : {len(pruned_idx)}  ({len(pruned_idx)/len(data_list)*100:.0f}%)")
print(f"  Score threshold           : {threshold:.4f}")
print(f"  Avg atoms retained        : {np.mean(retained_sizes):.2f}")
print(f"  Avg atoms pruned          : {np.mean(pruned_sizes):.2f}")
print(f"  Avg rings retained        : {np.mean(retained_rings):.2f}")
print(f"  Avg rings pruned          : {np.mean(pruned_rings):.2f}")
print()
print("  4 PNGs saved to visualizations/output/:")
print("    hp_A_score_distribution.png")
print("    hp_B_retained_vs_pruned_molecules.png")
print("    hp_C_structural_comparison.png")
print("    hp_D_graph_topology.png")
print()
print("  KEY INSIGHT:")
print("  Hard pruning keeps graphs that are STRUCTURALLY DIVERSE")
print("  (far from the average = high anomaly score).")
print("  Pruned graphs tend to be simpler/more common structures")
print("  that contribute less to learning.")
print("=" * 60)
