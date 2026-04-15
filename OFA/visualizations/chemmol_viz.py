"""
ChemMol dataset visualization
  Visual 1 — 12 real molecule structures (RDKit drawing)
  Visual 2 — Atom type + bond type distributions
  Visual 3 — How a molecule becomes a graph (node=atom, edge=bond)
  Visual 4 — Graph classification NOI construction (e2e_graph, lr_graph)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

import pandas as pd
import random; random.seed(42); np.random.seed(42)

print("=== ChemMol ===")

# load parquet directly (avoids HF path issues on Windows)
pq = os.path.normpath(os.path.join(OFA, "cache_data", "dataset",
     "molecule_property_instruction", "data",
     "chembl_pretraining-00000-of-00025-6a6e3f179bcd16a5.parquet"))
df = pd.read_parquet(pq)
smiles_list = df["graph"].dropna().tolist()
print(f"  Molecules in shard 0: {len(smiles_list)}")

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

# pick diverse molecules by size
valid_mols = [(s, Chem.MolFromSmiles(s)) for s in smiles_list if Chem.MolFromSmiles(s)]
valid_mols = [(s, m) for s, m in valid_mols if m is not None]
random.shuffle(valid_mols)
# sample across small/medium/large
small  = [(s,m) for s,m in valid_mols if m.GetNumAtoms() < 15][:4]
medium = [(s,m) for s,m in valid_mols if 15 <= m.GetNumAtoms() < 30][:4]
large  = [(s,m) for s,m in valid_mols if m.GetNumAtoms() >= 30][:4]
showcase = small + medium + large
print(f"  Showcasing {len(showcase)} molecules (small/medium/large)")

# ── Visual 1: 12 real molecule structures ────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle("ChemMol — Real Molecule Structures from ChEMBL Pretraining\n"
             "(Top: small <15 atoms | Middle: medium 15-30 | Bottom: large 30+)",
             fontsize=13, fontweight="bold")

from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

for idx, (smi, mol) in enumerate(showcase):
    ax = axes[idx // 4][idx % 4]
    try:
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        ax.imshow(img)
    except Exception:
        ax.text(0.5, 0.5, smi[:30], ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    ax.set_title(f"Atoms: {n_atoms}  Bonds: {n_bonds}  Rings: {n_rings}\n{smi[:35]}",
                 fontsize=7)
    ax.axis("off")

save(fig, "chemmol_1_molecule_structures.png")

# ── Visual 2: Atom + bond type distributions ──────────────────────────────────
from collections import Counter
atom_counter = Counter()
bond_counter = Counter()
hyb_counter  = Counter()
ring_counter = Counter()

for _, mol in valid_mols[:500]:
    for atom in mol.GetAtoms():
        atom_counter[atom.GetSymbol()] += 1
        hyb_counter[str(atom.GetHybridization()).split(".")[-1]] += 1
        ring_counter["in ring" if atom.IsInRing() else "not in ring"] += 1
    for bond in mol.GetBonds():
        bond_counter[str(bond.GetBondTypeAsDouble())] += 1

bond_names = {"1.0": "SINGLE", "1.5": "AROMATIC", "2.0": "DOUBLE", "3.0": "TRIPLE"}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("ChemMol — Atom & Bond Feature Distributions (500 molecule sample)",
             fontsize=12, fontweight="bold")

# 2a: top 15 atoms
ax = axes[0][0]
top_atoms = atom_counter.most_common(15)
a_labels, a_counts = zip(*top_atoms)
colors_a = plt.cm.Set3(np.linspace(0, 1, len(a_labels)))
ax.bar(a_labels, a_counts, color=colors_a, edgecolor="white")
ax.set_title("Atom type distribution\n(top 15 elements)", fontsize=10)
ax.set_ylabel("# atoms", fontsize=9)
ax.tick_params(axis='x', labelsize=9)

# 2b: bond types
ax = axes[0][1]
b_labels = [bond_names.get(k, k) for k in bond_counter.keys()]
b_counts = list(bond_counter.values())
colors_b = ["#64B5F6", "#81C784", "#FFB74D", "#E57373"][:len(b_labels)]
bars = ax.bar(b_labels, b_counts, color=colors_b, edgecolor="white")
ax.set_title("Bond type distribution", fontsize=10)
ax.set_ylabel("# bonds", fontsize=9)
for bar, c in zip(bars, b_counts):
    ax.text(bar.get_x() + bar.get_width()/2, c + 10, str(c), ha="center", fontsize=9)

# 2c: hybridization
ax = axes[1][0]
h_labels = list(hyb_counter.keys())
h_counts = list(hyb_counter.values())
ax.pie(h_counts, labels=h_labels, autopct="%1.1f%%",
       colors=plt.cm.Pastel1(np.linspace(0, 1, len(h_labels))), startangle=90)
ax.set_title("Atom hybridization types\n(SP, SP2, SP3, etc.)", fontsize=10)

# 2d: in-ring vs not
ax = axes[1][1]
r_labels = list(ring_counter.keys())
r_counts = list(ring_counter.values())
ax.pie(r_counts, labels=r_labels, autopct="%1.1f%%",
       colors=["#A5D6A7", "#EF9A9A"], startangle=90)
ax.set_title("Atoms in ring vs not\n(rings = aromatic/cyclic structures)", fontsize=10)

save(fig, "chemmol_2_feature_distributions.png")

# ── Visual 3: molecule → graph conversion ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("ChemMol — How a Molecule Becomes a Graph (SMILES → PyG Data)",
             fontsize=12, fontweight="bold")

# pick aspirin-like molecule
ex_smi = "CC(=O)Oc1ccccc1C(=O)O"   # aspirin
ex_mol = Chem.MolFromSmiles(ex_smi)
if ex_mol is None:
    ex_smi, ex_mol = showcase[1]

# 3a: 2D structure
ax = axes[0]
try:
    AllChem.Compute2DCoords(ex_mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(350, 350)
    drawer.DrawMolecule(ex_mol)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    ax.imshow(img)
except Exception:
    ax.text(0.5, 0.5, "molecule", ha="center", va="center", transform=ax.transAxes)
ax.set_title(f"2D molecular structure\nSMILES: {ex_smi}", fontsize=9)
ax.axis("off")

# 3b: as graph
ax = axes[1]
G_mol = nx.Graph()
atom_labels = {}
atom_colors = []
color_map = {"C": "#64B5F6", "O": "#EF5350", "N": "#4CAF50",
             "S": "#FFEB3B", "F": "#CE93D8"}
for atom in ex_mol.GetAtoms():
    idx = atom.GetIdx()
    sym = atom.GetSymbol()
    G_mol.add_node(idx)
    atom_labels[idx] = f"{sym}{idx}"
    atom_colors.append(color_map.get(sym, "#B0BEC5"))
bond_edge_colors = []
bond_widths = []
bond_type_map = {1.0: ("#64B5F6", 1.5), 1.5: ("#FF9800", 2.5),
                 2.0: ("#E91E63", 2.0), 3.0: ("#9C27B0", 2.5)}
for bond in ex_mol.GetBonds():
    bt = bond.GetBondTypeAsDouble()
    G_mol.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    c, w = bond_type_map.get(bt, ("#aaa", 1.0))
    bond_edge_colors.append(c); bond_widths.append(w)
pos_mol = nx.spring_layout(G_mol, seed=10)
nx.draw_networkx(G_mol, pos_mol, ax=ax, labels=atom_labels,
                 node_color=atom_colors, node_size=300, font_size=7,
                 edge_color=bond_edge_colors, width=bond_widths, alpha=0.9)
bond_legend = [mpatches.Patch(color="#64B5F6", label="single"),
               mpatches.Patch(color="#FF9800", label="aromatic"),
               mpatches.Patch(color="#E91E63", label="double")]
ax.legend(handles=bond_legend, fontsize=7)
ax.set_title("Graph representation\n(node=atom, edge=bond, color=element/bond type)", fontsize=9)
ax.axis("off")

# 3c: feature text for each atom
ax = axes[2]; ax.axis("off"); ax.set_facecolor("#F3E5F5")
sys.path.insert(0, os.path.join(OFA, "data/chemmol"))
from gen_raw_graph import atom_to_feature, bond_to_feature
feat_text = "REAL ATOM & BOND FEATURES (text → Sentence-BERT):\n\n"
for atom in list(ex_mol.GetAtoms())[:6]:
    feat_text += f"Atom {atom.GetIdx()} ({atom.GetSymbol()}):\n"
    feat_text += f"  {atom_to_feature(atom)[:80]}...\n\n"
feat_text += "\nBOND FEATURES:\n"
for bond in list(ex_mol.GetBonds())[:3]:
    feat_text += f"  {bond_to_feature(bond)[:70]}...\n"
ax.text(0.02, 0.98, feat_text, transform=ax.transAxes, fontsize=6.5,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("Node/edge feature text\n(unique texts deduplicated, indexed by ID)", fontsize=9)

save(fig, "chemmol_3_mol_to_graph.png")

# ── Visual 4: graph classification NOI ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("ChemMol — Graph Classification: How NOI Works on Whole Molecules",
             fontsize=12, fontweight="bold")

# for molecules, NOI connects to ALL atoms (the whole molecule is the subgraph)
mol_nodes = list(range(ex_mol.GetNumAtoms()))[:10]
noi_id    = 500
class_ids = [600 + i for i in range(4)]
mol_edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
             for b in ex_mol.GetBonds()
             if b.GetBeginAtomIdx() < 10 and b.GetEndAtomIdx() < 10]

draw_noi_graph(axes[0], mol_nodes, noi_id, class_ids, mol_edges,
    title="e2e_graph\nEnd-to-End Graph Classification",
    task_desc="Does this molecule have property X?\nNOI connects to ALL atoms.\nClass nodes = property labels.\n(e.g. HIV inhibitor? Toxic?)",
    class_labels=["prop_A", "prop_B", "prop_C", "prop_D"])

draw_noi_graph(axes[1], mol_nodes, noi_id, [], mol_edges,
    title="lr_graph\nFew-Shot Graph Classification",
    task_desc="Does this molecule belong to the\nsame class as support molecules?\nNo class nodes — GNN compares\nquery vs support molecule embeddings.")

save(fig, "chemmol_4_graph_classification_noi.png")
print("ChemMol done.\n")
