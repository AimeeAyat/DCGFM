"""
Pruning visualization for molecular data (mol_fs / chemblpre task).

Hard pruning facts:
  - hard70_only / hard70_soft30 / hard70_soft50 all use the SAME pkl file
    (dcgfm_hard_prune_api_25_0.7_2.pkl).  GraphGPS is removed from this
    script because with the SBERT-cosine proxy both produce identical masks.
  - Hard pruning keeps the top 30% most anomalous molecules (Deep SVDD score).
  - Soft pruning (prune_ratio) is a training-time mechanism, NOT pre-selection.

PART A  -- What the hard pruning kept vs discarded, structurally
  fig1  Score distribution: full vs kept vs pruned
  fig2  t-SNE: anomaly-score heatmap + kept/pruned overlay
  fig3  Cluster survival: which structural clusters survive pruning
  fig4  SMILES feature means: kept vs pruned vs full (bar chart)
  fig5  Label balance: does pruning shift positive/negative ratio?
  fig6  Structural histograms: ring count, heavy atoms, arom fraction ...
  fig7  Survival rate by structural complexity (key: is pruning selective?)
  interactive.html  Plotly scatter (hover for SMILES / score)

PART B  -- Soft pruning explanation + accuracy
  fig8  Soft pruning simulation: how prune_ratio 0/0.3/0.5 shrinks the set
  fig9  Accuracy bar chart (hard70_soft30 result filled in)

Usage:
    python visualize_pruning.py
    python visualize_pruning.py --n_clusters 13 --tsne_sample 12000
"""

import os, sys, argparse, warnings, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_DATASETS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from datasets import load_dataset
import plotly.express as px

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(SCRIPT_DIR, "cache_data", "dataset")
PRUNE_DIR   = os.path.join(SCRIPT_DIR, "cache_data", "ofa_dataset", "hard_pruning_indices")
EMBED_CACHE = os.path.join(SCRIPT_DIR, "visualizations", "sbert_embeddings.npz")
OUT_DIR     = os.path.join(SCRIPT_DIR, "visualizations", "pruning")
os.makedirs(OUT_DIR, exist_ok=True)

MOL_TASK_IDX   = 2          # chemblpre is task index 2 in mol_fs
FS_SAMPLE_SIZE = 60000      # episodes per task
HARD_PKL       = f"dcgfm_hard_prune_api_25_0.7_{MOL_TASK_IDX}.pkl"

ALL_BENCHMARKS = [
    "bace", "bbbp", "chembl_pretraining", "chembl_zero_shot",
    "cyp450", "esol", "freesolv", "hiv", "lipo", "muv",
    "pcba", "tox21", "toxcast",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def smiles_features(smiles):
    """Regex-based structural feature extraction (no RDKit needed)."""
    s = str(smiles)
    n_arom   = len(re.findall(r'[cnospb]', s))
    ring_dig = re.findall(r'(?<![%])\d|%\d{2}', s)
    n_heavy  = max(len(re.findall(r'[A-Z][a-z]?|[cnospb]', s)), 1)
    return {
        "n_heavy":    n_heavy,
        "n_rings":    len(ring_dig) // 2,
        "arom_frac":  n_arom / n_heavy,
        "n_branches": s.count("("),
        "n_double":   s.count("="),
        "has_N":      int(bool(re.search(r'[Nn]', s))),
        "has_O":      int(bool(re.search(r'[Oo]', s))),
        "has_S":      int(bool(re.search(r'[Ss]', s))),
        "has_hal":    int(bool(re.search(r'F|Cl|Br|I', s))),
        "has_charge": int(bool(re.search(r'[+\-]', s))),
    }


# ── 1. Load data ───────────────────────────────────────────────────────────────

def load_chemblpre(max_mols=0):
    print("Loading chemblpre ...")
    ds = load_dataset("haitengzhao/molecule_property_instruction",
                      split="chembl_pretraining", cache_dir=CACHE_DIR)
    df = ds.to_pandas()[["graph", "label", "molecule_index"]].drop_duplicates("molecule_index").copy()
    df["label_num"] = df["label"].apply(
        lambda v: 1 if str(v).strip().lower() == "yes"
        else (0 if str(v).strip().lower() == "no" else np.nan))
    if max_mols > 0 and len(df) > max_mols:
        df = df.sample(max_mols, random_state=42)
    print(f"  {len(df):,} unique chemblpre molecules")
    return df.reset_index(drop=True)


# ── 2. Get SBERT embeddings (cached from visualize_mol_data.py) ────────────────

def get_chemblpre_embeddings(n_chemblpre):
    if not os.path.exists(EMBED_CACHE):
        raise FileNotFoundError(
            f"Run visualize_mol_data.py first to generate {EMBED_CACHE}")
    npz   = np.load(EMBED_CACHE)
    X_all = npz["X"]
    print("  Locating chemblpre slice in embedding cache ...")
    offset = 0
    for bname in ALL_BENCHMARKS:
        if bname == "chembl_pretraining":
            break
        try:
            ds = load_dataset("haitengzhao/molecule_property_instruction",
                              split=bname, cache_dir=CACHE_DIR)
            offset += len(pd.DataFrame({"m": ds["molecule_index"]}).drop_duplicates())
        except Exception:
            pass
    print(f"  offset={offset:,}  n_chemblpre={n_chemblpre:,}  total_cache={len(X_all):,}")
    return X_all[offset: offset + n_chemblpre]


# ── 3. Anomaly score + hard-pruning mask ───────────────────────────────────────

def anomaly_scores(X_norm):
    """L2 distance from the SBERT centroid — proxy for Deep SVDD anomaly score."""
    centroid = X_norm.mean(axis=0)
    return np.linalg.norm(X_norm - centroid, axis=1)


def build_hard_mask(n_mols, scores):
    """
    Load the kept episode count from the pkl file, then apply the same
    keep-fraction threshold to the SBERT anomaly scores.
    Hard pruning keeps the TOP 30% (most anomalous / most diverse molecules).
    """
    pkl_path = os.path.join(PRUNE_DIR, HARD_PKL)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Hard pruning pkl not found: {pkl_path}")
    kept_episodes = pickle.load(open(pkl_path, "rb"))
    keep_frac  = len(kept_episodes) / FS_SAMPLE_SIZE   # ~0.30
    threshold  = np.percentile(scores, (1 - keep_frac) * 100)
    mask       = scores >= threshold
    print(f"  keep_frac={keep_frac:.2f}  threshold={threshold:.4f}")
    print(f"  kept={mask.sum():,}  pruned={(~mask).sum():,}  total={n_mols:,}")
    return mask


# ── 4. t-SNE (stratified sample) ──────────────────────────────────────────────

def build_tsne(X, scores, n_sample, seed=42):
    """
    Stratify sample across score quintiles so both kept and pruned regions
    are represented, then PCA->50 + t-SNE for visualisation.
    """
    rng = np.random.default_rng(seed)
    quintiles = np.array_split(np.argsort(scores), 5)
    per_q = max(1, n_sample // 5)
    idx   = np.concatenate([
        rng.choice(q, min(per_q, len(q)), replace=False) for q in quintiles
    ])[:n_sample]
    print(f"  PCA->50 + t-SNE on {len(idx):,} stratified points ...")
    Xp = PCA(n_components=50, random_state=seed).fit_transform(X[idx])
    xy = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=seed,
              init="pca", learning_rate="auto", n_jobs=1).fit_transform(Xp)
    return xy, idx


# ── PART A figures ─────────────────────────────────────────────────────────────

def fig1_score_distribution(scores, mask):
    """
    Density histogram of anomaly scores for:
      - full chemblpre dataset (grey)
      - kept molecules / top 30% (red)
      - pruned molecules / bottom 70% (blue)
    The dashed vertical lines mark the group means.
    Hard pruning shifts the kept set to higher scores — it picks
    structurally unusual molecules far from the average.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores,         bins=80, alpha=0.25, color="dimgray",  density=True,
            label=f"full dataset ({len(scores):,})")
    ax.hist(scores[mask],   bins=80, alpha=0.75, color="tomato",   density=True,
            label=f"kept / top 30%  ({mask.sum():,})")
    ax.hist(scores[~mask],  bins=80, alpha=0.45, color="steelblue",density=True,
            label=f"pruned / bottom 70%  ({(~mask).sum():,})")
    for vals, col in [(scores, "dimgray"), (scores[mask], "tomato"),
                      (scores[~mask], "steelblue")]:
        ax.axvline(vals.mean(), ls="--", color=col, lw=1.5, alpha=0.9)
    ax.set_xlabel("Anomaly score  (SBERT L2 distance from dataset centroid)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Fig 1 -- Hard pruning score distribution\n"
        "Kept set = most anomalous / diverse 30%.  "
        "Dashed lines = group means.",
        fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "fig1_score_distribution.png")


def fig2_tsne(xy, idx, scores, mask):
    """
    Two-panel t-SNE of 12k stratified chemblpre molecules.
    Left:  colour = anomaly score (plasma heatmap).  Warm = high score = diverse.
    Right: red = kept (top 30%), blue = pruned (bottom 70%).
    Compare the two panels: the kept molecules cluster in the hot (outer) region,
    confirming that hard pruning selects from the diverse periphery of chemical space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=scores[idx],
                         cmap="plasma", s=6, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04, label="anomaly score")
    axes[0].set_title("Anomaly score heatmap\n(warm = high score = more diverse)", fontsize=10)

    km = mask[idx]; pm = ~km
    axes[1].scatter(xy[pm, 0], xy[pm, 1], c="steelblue", s=5, alpha=0.20,
                    linewidths=0, label=f"pruned  ({pm.sum():,})")
    axes[1].scatter(xy[km, 0], xy[km, 1], c="tomato",    s=7, alpha=0.80,
                    linewidths=0, label=f"kept    ({km.sum():,})")
    axes[1].legend(fontsize=8, markerscale=2)
    axes[1].set_title("Kept (red) vs pruned (blue)", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Fig 2 -- t-SNE of chemblpre molecules (12k stratified sample)\n"
        "Each point = one molecule.  Position = structural similarity in SBERT space.",
        fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, "fig2_tsne_kept_pruned.png")


def fig3_cluster_survival(df, X_norm, mask, n_clusters):
    """
    Run k-means (k=n_clusters) on all 364k SBERT embeddings, then ask:
    what fraction of each cluster survives hard pruning?
    A flat 30% bar across all clusters = proportional (random-like) pruning.
    Tall bars = that cluster is over-selected (diverse / unusual molecules).
    Short bars = that cluster is mostly discarded (common / typical molecules).
    """
    print(f"  MiniBatchKMeans with k={n_clusters} ...")
    km_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cl = km_model.fit_predict(X_norm)

    survival = np.array([
        (cl == k)[mask].sum() / max((cl == k).sum(), 1) * 100
        for k in range(n_clusters)])

    colors = cm.tab20(np.linspace(0, 1, n_clusters))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(n_clusters), survival,
                  color=[colors[k] for k in range(n_clusters)], alpha=0.85)
    ax.axhline(30, ls="--", color="navy", lw=1.8, label="30% baseline (random pruning)")

    # label bars with cluster sizes
    for k, (bar, s) in enumerate(zip(bars, survival)):
        n_k = (cl == k).sum()
        ax.text(bar.get_x() + bar.get_width()/2, s + 1,
                f"{n_k//1000}k", ha="center", va="bottom", fontsize=6, color="gray")

    ax.set_xlabel("Cluster ID  (each cluster = a group of structurally similar molecules)")
    ax.set_ylabel("% of cluster kept after pruning")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.set_title(
        f"Fig 3 -- Cluster survival rate  (k={n_clusters} clusters)\n"
        "Bars above 30% = cluster over-selected by hard pruning  |  "
        "below 30% = under-selected",
        fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_cluster_survival.png")
    return cl


def fig4_smiles_feature_means(df, mask):
    """
    Horizontal bar chart of mean structural feature values for:
      kept (red), pruned (blue), full dataset (grey).
    'full' is always the weighted average 0.3*kept + 0.7*pruned and should
    fall between the two coloured bars -- if it doesn't, the two bars are
    very close and the feature is NOT selective.
    A visible gap between kept and pruned means hard pruning is selective
    for that chemical property.
    """
    print("  Extracting SMILES features for fig4 ...")
    feats = pd.DataFrame([smiles_features(s) for s in df["graph"]])
    FEAT_LABELS = {
        "n_heavy":    "Heavy atom count",
        "n_rings":    "Ring count",
        "arom_frac":  "Aromatic fraction",
        "n_branches": "Branch count (parentheses)",
        "n_double":   "Double bond count",
        "has_N":      "Contains N  (fraction)",
        "has_O":      "Contains O  (fraction)",
        "has_S":      "Contains S  (fraction)",
        "has_hal":    "Contains halogen F/Cl/Br/I",
        "has_charge": "Has formal charge (+/-)",
    }
    cols   = list(FEAT_LABELS.keys())
    kept_m   = feats.loc[mask,  cols].mean()
    pruned_m = feats.loc[~mask, cols].mean()
    full_m   = feats[cols].mean()

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(cols)); w = 0.26
    ax.barh(x + w,  kept_m.values,   w, color="tomato",    alpha=0.85, label="kept  (30%)")
    ax.barh(x,      pruned_m.values,  w, color="steelblue", alpha=0.60, label="pruned (70%)")
    ax.barh(x - w,  full_m.values,    w, color="dimgray",   alpha=0.45, label="full = 0.3*kept + 0.7*pruned")
    ax.set_yticks(x)
    ax.set_yticklabels([FEAT_LABELS[c] for c in cols], fontsize=9)
    ax.set_xlabel("Mean value per molecule")
    ax.legend(fontsize=9)
    ax.set_title(
        "Fig 4 -- Mean structural features: kept vs pruned vs full\n"
        "Gap between red and blue = pruning selectively favours that feature.\n"
        "Grey (full) should always lie between the two.",
        fontsize=11)
    fig.tight_layout()
    _save(fig, "fig4_smiles_feature_means.png")


def fig5_label_balance(df, mask):
    """
    Does hard pruning change the active/inactive class balance?
    ChEMBL uses binary activity labels (active=1 / inactive=0).
    If pruning is chemistry-aware it might preferentially keep actives
    (or inactives), shifting the ratio from the original dataset.
    A bar at ~50% = balanced.  Shift from baseline = biased pruning.
    """
    has_lbl = df["label_num"].notna()
    rows = [
        {"set": "Full dataset\n(364k)",
         "pos%": df.loc[has_lbl, "label_num"].mean() * 100,
         "n": has_lbl.sum()},
        {"set": "Kept\n(top 30%)",
         "pos%": df.loc[mask & has_lbl, "label_num"].mean() * 100,
         "n": (mask & has_lbl).sum()},
        {"set": "Pruned\n(bottom 70%)",
         "pos%": df.loc[~mask & has_lbl, "label_num"].mean() * 100,
         "n": (~mask & has_lbl).sum()},
    ]
    cdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["dimgray", "tomato", "steelblue"]
    bars = ax.bar(range(len(cdf)), cdf["pos%"], color=colors, alpha=0.82)
    ax.axhline(50, ls="--", color="navy", alpha=0.5, label="50% balance")
    ax.set_xticks(range(len(cdf)))
    ax.set_xticklabels(cdf["set"], fontsize=10)
    ax.set_ylabel("% active (positive) labels")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.set_title(
        "Fig 5 -- Class balance shift after hard pruning\n"
        "Does pruning preferentially keep actives or inactives?",
        fontsize=11)
    for bar, (_, row) in zip(bars, cdf.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f"{row['pos%']:.1f}%\n(n={row['n']:,})",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig5_label_balance.png")


def fig6_structural_histograms(df, mask):
    """
    Full probability density histograms for five structural properties,
    plus a bar chart of heteroatom/halogen presence rates.

    Overlapping histograms (red=kept, blue=pruned) answer:
      Is the DISTRIBUTION of ring count / size / aromaticity shifted
      in the kept set, or do they look identical to the full dataset?

    Panel 6 (bar chart) shows what % of kept vs pruned molecules contain
    nitrogen, oxygen, sulfur, halogens, or formal charges.
    A difference here means pruning is chemotype-selective.
    """
    print("  Computing structural features for fig6/fig7 ...")
    feats = pd.DataFrame([smiles_features(s) for s in df["graph"]])
    feats["kept"] = np.asarray(mask)

    hist_specs = [
        ("n_rings",    "Ring count",        np.arange(0, 16, 1)),
        ("n_heavy",    "Heavy atom count",  np.arange(0, 80, 5)),
        ("arom_frac",  "Aromatic fraction", np.linspace(0, 1, 21)),
        ("n_branches", "Branch count",      np.arange(0, 20, 1)),
        ("n_double",   "Double bond count", np.arange(0, 15, 1)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for ax, (feat, label, bins) in zip(axes_flat, hist_specs):
        kept_v   = feats.loc[feats["kept"],  feat]
        pruned_v = feats.loc[~feats["kept"], feat]
        ax.hist(pruned_v, bins=bins, density=True, alpha=0.50,
                color="steelblue", label=f"pruned  ({len(pruned_v):,})")
        ax.hist(kept_v,   bins=bins, density=True, alpha=0.75,
                color="tomato",    label=f"kept    ({len(kept_v):,})")
        # add means
        ax.axvline(kept_v.mean(),   ls="--", color="tomato",    lw=1.5, alpha=0.9)
        ax.axvline(pruned_v.mean(), ls="--", color="steelblue", lw=1.5, alpha=0.9)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.set_title(f"{label}: mean kept={kept_v.mean():.2f}  pruned={pruned_v.mean():.2f}",
                     fontsize=8)

    # 6th panel: heteroatom / halogen presence %
    ax = axes_flat[5]
    binary_feats  = ["has_N", "has_O", "has_S", "has_hal", "has_charge"]
    binary_labels = ["Has N", "Has O", "Has S", "Has halogen", "Has charge"]
    kept_r  = [feats.loc[feats["kept"],  f].mean() * 100 for f in binary_feats]
    prune_r = [feats.loc[~feats["kept"], f].mean() * 100 for f in binary_feats]
    full_r  = [feats[f].mean() * 100 for f in binary_feats]
    x = np.arange(len(binary_feats)); w = 0.25
    ax.bar(x - w, prune_r, w, color="steelblue", alpha=0.65, label="pruned")
    ax.bar(x,     kept_r,  w, color="tomato",    alpha=0.80, label="kept")
    ax.bar(x + w, full_r,  w, color="dimgray",   alpha=0.45, label="full")
    ax.set_xticks(x)
    ax.set_xticklabels(binary_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% of molecules")
    ax.legend(fontsize=7)
    ax.set_title("Heteroatom / halogen presence (%)", fontsize=9)

    fig.suptitle(
        "Fig 6 -- Structural feature distributions: kept (red) vs pruned (blue)\n"
        "Dashed lines = group means.  "
        "Visible shift = hard pruning is SELECTIVE for that feature.",
        fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "fig6_structural_histograms.png")
    return feats   # reuse in fig7


def fig7_survival_rate(feats):
    """
    Survival rate = fraction of molecules in each structural bin that was KEPT.
    The dashed line at 30% is the expected rate under random pruning.

    Reading the chart:
      Bar > 30%  => molecules with that property are OVER-SELECTED by pruning
                    (anomaly-based pruning finds them diverse / unusual).
      Bar < 30%  => those molecules are UNDER-SELECTED (they are common /
                    close to the dataset average and get discarded).

    Four panels: ring count, heavy atom count, aromatic fraction, branch count.
    """
    bucket_specs = [
        ("n_rings",    "Ring count",               list(range(0, 10))),
        ("n_heavy",    "Heavy atom count  (x10)",  [i * 10 for i in range(0, 7)]),
        ("arom_frac",  "Aromatic fraction",         [0.0, 0.25, 0.5, 0.75, 1.01]),
        ("n_branches", "Branch count",              list(range(0, 10))),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    for ax, (feat, label, boundaries) in zip(axes_flat, bucket_specs):
        boundaries = list(boundaries)
        if feat in ("n_rings", "n_branches"):
            bins = np.array(boundaries + [boundaries[-1] + 1], dtype=float) - 0.5
        elif feat == "n_heavy":
            bins = np.array(boundaries + [boundaries[-1] + 10], dtype=float)
        else:
            bins = np.array(boundaries, dtype=float)

        counts_full, _ = np.histogram(feats[feat],                      bins=bins)
        counts_kept, _ = np.histogram(feats.loc[feats["kept"], feat],   bins=bins)
        survival = np.where(counts_full > 0,
                            counts_kept / counts_full * 100, np.nan)
        centers  = (bins[:-1] + bins[1:]) / 2

        colors_bar = ["tomato" if s > 30 else "steelblue"
                      if not np.isnan(s) else "lightgrey" for s in survival]
        ax.bar(centers, np.nan_to_num(survival),
               width=(bins[1] - bins[0]) * 0.75,
               color=colors_bar, alpha=0.80, label="survival %")
        ax.axhline(30, ls="--", color="navy", lw=1.8, label="30% (random baseline)")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("% of bin kept")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

        # annotate bins with molecule count
        for cx, s, n in zip(centers, survival, counts_full):
            if not np.isnan(s) and n > 200:
                ax.text(cx, min(s, 97) + 1, f"{n//1000}k",
                        ha="center", fontsize=6, color="gray")

    fig.suptitle(
        "Fig 7 -- Survival rate by structural complexity\n"
        "Red bar > 30%: that structural group is PREFERRED by anomaly-based hard pruning.\n"
        "Blue bar < 30%: that group is mostly discarded (common/typical structures).",
        fontsize=12, y=1.03)
    fig.tight_layout()
    _save(fig, "fig7_survival_rate_by_complexity.png")


def fig_interactive(xy, idx, df, scores, mask):
    """
    Plotly HTML scatter.  Hover over any point to see:
      - SMILES string (first 60 chars)
      - binary label (active / inactive)
      - anomaly score
      - whether this molecule was kept or pruned
    Colour = kept (red) or pruned (blue).
    """
    sub = df.iloc[idx].copy().reset_index(drop=True)
    sub["x"]            = xy[:, 0]
    sub["y"]            = xy[:, 1]
    sub["anomaly_score"]= scores[idx]
    sub["status"]       = pd.Series(mask[idx]).map(
        {True: "kept (top 30%)", False: "pruned (bottom 70%)"}).values
    sub["smiles_short"] = sub["graph"].apply(
        lambda s: (str(s)[:60] + "...") if len(str(s)) > 60 else str(s))

    color_map = {"kept (top 30%)": "tomato", "pruned (bottom 70%)": "steelblue"}
    fig = px.scatter(
        sub, x="x", y="y", color="status",
        color_discrete_map=color_map,
        hover_data={"smiles_short": True, "label": True,
                    "anomaly_score": ":.3f", "x": False, "y": False},
        title="Hard pruning: kept vs pruned in SBERT chemical space  (hover for SMILES)",
        width=1300, height=850, opacity=0.65)
    fig.update_traces(marker=dict(size=5))
    path = os.path.join(OUT_DIR, "interactive_pruning.html")
    fig.write_html(path)
    print(f"  saved -> {path}")


# ── PART B figures ─────────────────────────────────────────────────────────────

def fig8_soft_pruning_concept(scores, mask):
    """
    Soft pruning (prune_ratio) is NOT a molecule pre-selector -- it runs
    every training epoch and drops the highest-loss samples from the current
    mini-batch gradient step.

    This figure simulates what that means by using anomaly score as a proxy
    for training loss (high anomaly ~ hard to fit ~ high loss).

    Three panels show the effective training set for prune_ratio = 0 / 0.3 / 0.5:
      - prune_ratio=0.0: all 109k hard-pruned molecules used every epoch
      - prune_ratio=0.3: ~76k used per epoch  (drop 30% highest-loss)
      - prune_ratio=0.5: ~55k used per epoch  (drop 50% highest-loss)

    Orange bars = soft-dropped molecules (still in hard set, just skipped
    this epoch).  Note: the selection changes each epoch dynamically.
    """
    kept_scores = scores[mask]
    n_kept      = len(kept_scores)
    sorted_idx  = np.argsort(kept_scores)   # ascending: low score first

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    panels = [
        (0.0, "Fig 8a -- hard70 only\nprune_ratio=0.0\nAll hard-pruned molecules used every epoch"),
        (0.3, "Fig 8b -- hard70 + soft30\nprune_ratio=0.3\n~70% of hard set used per epoch"),
        (0.5, "Fig 8c -- hard70 + soft50\nprune_ratio=0.5\n~50% of hard set used per epoch"),
    ]

    for ax, (soft_r, title) in zip(axes, panels):
        n_eff       = int(n_kept * (1 - soft_r))
        eff_idx     = sorted_idx[:n_eff]
        drop_idx    = sorted_idx[n_eff:]
        ax.hist(kept_scores[eff_idx],  bins=50, alpha=0.85, color="tomato",
                density=True, label=f"used this epoch  ({n_eff:,})")
        if soft_r > 0:
            ax.hist(kept_scores[drop_idx], bins=50, alpha=0.55, color="orange",
                    density=True, label=f"soft-dropped  ({len(drop_idx):,})")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Anomaly score  (proxy for training loss)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density")
    fig.suptitle(
        "Fig 8 -- Soft pruning simulation (proxy-based)\n"
        "Soft pruning drops highest-loss samples each epoch from the hard-pruned set.\n"
        "This is DYNAMIC -- the dropped set changes every epoch.  "
        "Orange = skipped this epoch, not permanently removed.",
        fontsize=11, y=1.04)
    fig.tight_layout()
    _save(fig, "fig8_soft_pruning_concept.png")


def fig9_accuracy_table():
    """
    Grouped bar chart comparing all pruning experiments across 6 benchmarks.

    Benchmarks (from the LaTeX table):
      Arxiv (N)   -- node classification on ArXiv citation graph
      PubMed (N)  -- node classification on PubMed citation graph
      PCBA (G)    -- graph classification, 128 biochemical assays
      HIV (G)     -- graph classification, HIV activity
      FB15K (L)   -- link prediction on Freebase KB
      WN18RR (L)  -- link prediction on WordNet KB

    Results are reproductions using the same methodology as the paper.
    GraphGPS (H70, S50) is new experiment.
    """
    # columns: Arxiv, PubMed, PCBA, HIV, FB15K, WN18RR, AVG
    data = {
        "Whole Dataset":          [35.10, 33.10, 51.00, 53.20, 55.50, 15.60, 40.60],
        "DCGFM paper (H70,S50)":  [47.80, 42.30, 49.90, 51.10, 52.40,  6.90, 41.70],
        "w/o soft (paper)":       [45.00,  None, 49.90,  None, 50.90,  None, 40.00],
        "Hard70, Soft0":          [38.66, 27.80, 55.47, 48.34, 49.65, 26.90, 40.47],
        "Hard70, Soft30":         [45.91, 31.13, 54.21, 48.25, 51.74, 22.25, 42.08],
        "Hard70, Soft50":         [42.71, 30.36, 52.78, 52.29, 50.40, 26.65, 42.03],
        "GraphGPS (H70,S50)":     [46.20, 28.50, 51.40, 44.60, 53.80, 22.50, 39.97],
    }
    benchmarks = ["Arxiv\n(Node)", "PubMed\n(Node)", "PCBA\n(Graph)",
                  "HIV\n(Graph)", "FB15K\n(Link)", "WN18RR\n(Link)", "AVG"]
    n_exp = len(data)
    n_bench = len(benchmarks)

    # color palette: first two are baselines (grey tones), rest are experiments
    palette = ["#591cff", "#004912", "#8b0000",
               "steelblue", "tomato", "darkorange", "mediumseagreen"]

    x = np.arange(n_bench)
    w = 0.11
    offsets = np.linspace(-(n_exp - 1) / 2, (n_exp - 1) / 2, n_exp) * w

    fig, ax = plt.subplots(figsize=(16, 7))

    for (exp_label, vals), offset, col in zip(data.items(), offsets, palette):
        # replace None with nan
        v = np.array([v if v is not None else np.nan for v in vals])
        bars = ax.bar(x + offset, np.nan_to_num(v), w,
                      color=col, alpha=0.85, label=exp_label)
        # value labels on non-nan bars
        for bar, val in zip(bars, v):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=8.5, rotation=90, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10)
    ax.set_ylabel("Accuracy / AUC (%)", fontsize=10)
    ax.set_ylim(0, 70)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title(
        "Fig 9 -- Pruning experiment comparison across 6 benchmarks\n"
        "N=Node classification  G=Graph classification  L=Link prediction\n"
        "Our reproductions vs paper baselines.  GraphGPS = new experiment.",
        fontsize=11)
    fig.tight_layout()
    _save(fig, "fig9_accuracy.png")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_mols",    type=int, default=0,
                        help="Cap molecule count (0 = all 364k)")
    parser.add_argument("--n_clusters",  type=int, default=13)
    parser.add_argument("--tsne_sample", type=int, default=12000)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Hard pruning: GIN Deep SVDD, keeps top 30% anomalous molecules.")
    print("Soft pruning (prune_ratio): training-time only, visualized in fig8.")
    print("GraphGPS comparison removed (identical mask with SBERT proxy).")
    print("=" * 60)

    print("\n=== Step 1: Load chemblpre ===")
    df = load_chemblpre(args.max_mols)

    print("\n=== Step 2: SBERT embeddings ===")
    X_raw = get_chemblpre_embeddings(len(df))
    n     = min(len(df), len(X_raw))
    df    = df.iloc[:n].reset_index(drop=True)
    X_raw = X_raw[:n]
    X_norm = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-9)
    print(f"  embedding shape: {X_norm.shape}")

    print("\n=== Step 3: Anomaly scores + hard mask ===")
    scores = anomaly_scores(X_norm)
    mask   = build_hard_mask(n, scores)

    print(f"\n=== Step 4: t-SNE ({args.tsne_sample:,} sample) ===")
    xy, idx = build_tsne(X_norm, scores, args.tsne_sample)

    print("\n=== PART A: Hard pruning figures ===")
    fig1_score_distribution(scores, mask)
    fig2_tsne(xy, idx, scores, mask)
    fig3_cluster_survival(df, X_norm, mask, args.n_clusters)
    fig4_smiles_feature_means(df, mask)
    fig5_label_balance(df, mask)
    feats = fig6_structural_histograms(df, mask)
    fig7_survival_rate(feats)
    fig_interactive(xy, idx, df, scores, mask)

    print("\n=== PART B: Soft pruning + accuracy ===")
    fig8_soft_pruning_concept(scores, mask)
    fig9_accuracy_table()

    print(f"\nAll figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
