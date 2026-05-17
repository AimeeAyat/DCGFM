"""
mol_hop_analysis.py — Graph-diameter (hop) distribution for molecular datasets.

For each molecule we compute the diameter = longest shortest path between any two
atoms.  This is the minimum GNN depth needed to propagate information across the
whole molecule.  A 2-layer GIN only sees 2-hop neighbourhoods; molecules whose
diameter > 2 will have atoms that never "see" each other.

Datasets analysed: chemblpre, chemhiv, chempcba
Run from OFA/ directory:
    python mol_hop_analysis.py
"""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

# ── Config ──────────────────────────────────────────────────────────────────
CACHE_DIR   = "./cache_data/dataset"
OUT_DIR     = "./analysis_output"
DIAM_CACHE  = os.path.join(OUT_DIR, "mol_diameters.pkl")
# chemblpre has 360 k+ molecules — sample for speed; set 0 to use all
SAMPLE_CAP  = {"chemblpre": 60_000, "chemhiv": 0, "chempcba": 0}

NAME_TO_SPLIT = {
    "chemblpre": "chembl_pretraining",
    "chempcba":  "pcba",
    "chemhiv":   "hiv",
}
DATASETS = ["chemblpre", "chemhiv", "chempcba"]

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def smiles_to_nx(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    G = nx.Graph()
    G.add_nodes_from(range(mol.GetNumAtoms()))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


def graph_diameter(G):
    """Diameter of the largest connected component; None for empty graphs."""
    if G is None or G.number_of_nodes() == 0:
        return None
    if G.number_of_nodes() == 1:
        return 0
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        return nx.diameter(G)
    except Exception:
        return None


# ── Load / compute diameters ─────────────────────────────────────────────────

def compute_diameters(name, smiles_list):
    cap = SAMPLE_CAP.get(name, 0)
    if cap and len(smiles_list) > cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(smiles_list), cap, replace=False)
        smiles_list = [smiles_list[i] for i in idx]
        print(f"  Sampled {cap:,} / {len(smiles_list):,} molecules")

    diameters = []
    skipped   = 0
    for i, smi in enumerate(smiles_list):
        if i % 10_000 == 0:
            print(f"  {i:>7,} / {len(smiles_list):,}  "
                  f"(valid so far: {len(diameters):,})", flush=True)
        G = smiles_to_nx(smi)
        d = graph_diameter(G)
        if d is None:
            skipped += 1
        else:
            diameters.append(d)

    print(f"  Done — {len(diameters):,} valid, {skipped} skipped")
    return np.array(diameters, dtype=np.int32)


def load_smiles(name):
    import os as _os
    _os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    print(f"  Loading {NAME_TO_SPLIT[name]} from HF cache …")
    ds = load_dataset(
        "haitengzhao/molecule_property_instruction",
        split=NAME_TO_SPLIT[name],
        cache_dir=CACHE_DIR,
    )
    smiles = [row for row in ds["graph"] if row is not None]
    smiles = list(dict.fromkeys(smiles))   # deduplicate, preserve order
    print(f"  {len(smiles):,} unique SMILES")
    return smiles


def get_all_diameters():
    if os.path.exists(DIAM_CACHE):
        print(f"Loading cached diameters from {DIAM_CACHE}")
        with open(DIAM_CACHE, "rb") as f:
            return pickle.load(f)

    all_d = {}
    for name in DATASETS:
        print(f"\n{'='*50}\n{name}")
        smiles = load_smiles(name)
        all_d[name] = compute_diameters(name, smiles)

    with open(DIAM_CACHE, "wb") as f:
        pickle.dump(all_d, f)
    print(f"\nDiameters cached to {DIAM_CACHE}")
    return all_d


# ── Plot ─────────────────────────────────────────────────────────────────────

# Colour thresholds that map onto GNN depth milestones
THRESHOLDS = [
    (2,  "tomato",      "≤ 2 hops\n(2-layer GIN covers fully)"),
    (4,  "darkorange",  "≤ 4 hops"),
    (6,  "gold",        "≤ 6 hops"),
    (10, "steelblue",   "≤ 10 hops"),
]


def bar_colour(x):
    for t, c, _ in THRESHOLDS:
        if x <= t:
            return c
    return "slategray"


def plot_dataset(ax, name, d):
    p99 = int(np.percentile(d, 99))
    max_bin = max(p99 + 1, 12)
    bins = np.arange(-0.5, max_bin + 1.5, 1)

    counts, edges = np.histogram(d, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    colours = [bar_colour(int(round(c))) for c in centers]
    ax.bar(centers, counts, width=0.85, color=colours, edgecolor="white", linewidth=0.4)

    # Vertical reference lines
    for t, c, _ in THRESHOLDS:
        if t <= max_bin:
            ax.axvline(t, color=c, linestyle="--", linewidth=1.2, alpha=0.8)

    # Cumulative % annotations at key thresholds
    for t, c, _ in THRESHOLDS:
        if t <= max_bin:
            pct = (d <= t).mean() * 100
            y_pos = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else counts.max()
            ax.text(t, counts.max() * 0.97, f"{pct:.0f}%",
                    ha="center", va="top", fontsize=7, color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor=c))

    ax.set_title(f"{name}  (n={len(d):,})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Graph diameter  (hops)", fontsize=9)
    ax.set_ylabel("# molecules", fontsize=9)
    ax.set_xlim(-0.5, max_bin + 0.5)
    ax.tick_params(axis="both", labelsize=8)

    # Stats box
    stats = (f"min={d.min()}  max={d.max()}\n"
             f"mean={d.mean():.1f}  median={int(np.median(d))}")
    ax.text(0.97, 0.97, stats, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))


def make_plot(all_d):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, name in zip(axes, DATASETS):
        plot_dataset(ax, name, all_d[name])

    # Shared legend
    patches = [mpatches.Patch(color=c, label=lbl) for _, c, lbl in THRESHOLDS]
    patches.append(mpatches.Patch(color="slategray", label="> 10 hops"))
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Molecular graph diameter distribution\n"
        "Diameter = longest shortest path = minimum GNN layers to reach all atoms",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "mol_hop_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    return out


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_d):
    header = f"\n{'Dataset':<12} {'n':>7} {'min':>4} {'max':>4} {'mean':>6} {'med':>4}  " \
             f"{'≤2':>6} {'≤4':>6} {'≤6':>6} {'≤10':>6} {'>10':>6}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for name in DATASETS:
        d = all_d[name]
        print(
            f"{name:<12} {len(d):>7,} {d.min():>4} {d.max():>4} "
            f"{d.mean():>6.1f} {int(np.median(d)):>4}  "
            f"{(d<=2).mean()*100:>5.1f}% "
            f"{(d<=4).mean()*100:>5.1f}% "
            f"{(d<=6).mean()*100:>5.1f}% "
            f"{(d<=10).mean()*100:>5.1f}% "
            f"{(d>10).mean()*100:>5.1f}%"
        )
    print("=" * len(header))
    print("\nInterpretation:")
    print("  ≤2 hops  → a 2-layer GIN already covers the full molecule")
    print("  3-4 hops → need ≥ 3-4 layers; standard GIN depth is often sufficient")
    print("  ≥ 5 hops → long-range; deeper GIN or attention-based model (GPS, GT) recommended")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_d = get_all_diameters()
    print_summary(all_d)
    make_plot(all_d)
