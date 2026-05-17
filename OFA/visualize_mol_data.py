"""
Molecular dataset exploration: SBERT embeddings + t-SNE + MiniBatchKMeans.

Strategy for large data (1M+ molecules):
  - K-Means     : MiniBatchKMeans — runs on ALL molecules, O(n) memory
  - t-SNE       : PCA 768→50 first, then stratified sample of tsne_sample
                  points (default 15000). t-SNE is a visualisation tool and
                  cannot handle 1M points — sampling is correct practice.
  - All 6 static figures and 1 interactive HTML use the t-SNE sample, but
    the cluster assignments come from the full-data MiniBatchKMeans.

Usage:
    python visualize_mol_data.py                        # all data, 15k t-SNE sample
    python visualize_mol_data.py --tsne_sample 30000    # bigger t-SNE sample
    python visualize_mol_data.py --n_clusters 20
    python visualize_mol_data.py --benchmarks hiv bace bbbp tox21
    python visualize_mol_data.py --no_cache             # force re-encode
"""

import os, argparse, warnings
os.environ["HF_DATASETS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import plotly.express as px

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(SCRIPT_DIR, "cache_data", "dataset")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "cache_data", "model", "st_model_plain")
OUT_DIR     = os.path.join(SCRIPT_DIR, "visualizations")
EMBED_CACHE = os.path.join(OUT_DIR, "sbert_embeddings.npz")

ALL_BENCHMARKS = [
    "bace", "bbbp", "chembl_pretraining", "chembl_zero_shot",
    "cyp450", "esol", "freesolv", "hiv", "lipo", "muv",
    "pcba", "tox21", "toxcast",
]

TASK_TYPE = {   # regression benchmarks (continuous labels, no Yes/No)
    "esol": "regression", "freesolv": "regression", "lipo": "regression",
}

os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Load data ───────────────────────────────────────────────────────────────

def load_benchmarks(benchmarks, max_per_split):
    rows = []
    for bname in benchmarks:
        print(f"  loading {bname} …", end=" ", flush=True)
        try:
            ds = load_dataset(
                "haitengzhao/molecule_property_instruction",
                split=bname, cache_dir=CACHE_DIR,
            )
        except Exception as e:
            print(f"SKIP ({e})"); continue

        df = ds.to_pandas()[["graph", "label", "split", "molecule_index"]].copy()
        df["benchmark"] = bname
        df = df.drop_duplicates(subset=["molecule_index"]).copy()

        if max_per_split > 0 and len(df) > max_per_split:
            df = df.sample(max_per_split, random_state=42)

        def parse_label(v):
            if str(v).strip().lower() == "yes":  return 1
            if str(v).strip().lower() == "no":   return 0
            try:    return float(v)
            except: return np.nan

        df["label_num"] = df["label"].apply(parse_label)
        rows.append(df)
        print(f"{len(df)} molecules")

    if not rows:
        raise RuntimeError("No benchmarks loaded.")
    return pd.concat(rows, ignore_index=True)


# ── 2. SBERT encoding ──────────────────────────────────────────────────────────

def encode_smiles(smiles_list, batch_size=256, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEncoding {len(smiles_list):,} SMILES on {device} …")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            enc   = tokenizer(batch, padding=True, truncation=True,
                              max_length=128, return_tensors="pt").to(device)
            out   = model(**enc)
            mask  = enc["attention_mask"].unsqueeze(-1).float()
            emb   = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_embs.append(emb.cpu().float().numpy())
            if (i // batch_size) % 20 == 0:
                pct = (i + len(batch)) / len(smiles_list) * 100
                print(f"  {i+len(batch):>8,} / {len(smiles_list):,}  ({pct:.1f}%)", end="\r")
    print()
    return np.vstack(all_embs)


# ── 3. MiniBatchKMeans on full data ────────────────────────────────────────────

def run_kmeans(X, n_clusters, seed=42):
    print(f"MiniBatchKMeans (k={n_clusters}) on {X.shape[0]:,} molecules …")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         batch_size=4096, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    try:
        idx = np.random.default_rng(seed).choice(len(X), min(8000, len(X)), replace=False)
        sil = silhouette_score(X[idx], labels[idx])
        print(f"  Silhouette (sample 8k): {sil:.4f}")
    except Exception:
        pass
    return labels, km


# ── 4. PCA + stratified t-SNE sample ──────────────────────────────────────────

def build_tsne_sample(X, df, cluster_labels, tsne_sample, seed=42):
    """
    Stratified sample: proportional to benchmark size, but each benchmark
    gets at least min(50, its_size) points so small ones are visible.
    Returns (X_sample, df_sample, cluster_sample, orig_indices).
    """
    rng = np.random.default_rng(seed)
    n_total = len(df)
    indices = []

    benchmarks = df["benchmark"].unique()
    # compute per-benchmark quota
    sizes   = {b: (df["benchmark"] == b).sum() for b in benchmarks}
    quotas  = {b: max(50, int(tsne_sample * sizes[b] / n_total)) for b in benchmarks}
    # scale down if over budget
    total_q = sum(quotas.values())
    if total_q > tsne_sample:
        scale = tsne_sample / total_q
        quotas = {b: max(10, int(q * scale)) for b, q in quotas.items()}

    for b in benchmarks:
        mask = np.where(df["benchmark"].values == b)[0]
        n    = min(quotas[b], len(mask))
        indices.extend(rng.choice(mask, n, replace=False).tolist())

    indices = np.array(indices)
    return X[indices], df.iloc[indices].reset_index(drop=True), cluster_labels[indices], indices


def run_pca_tsne(X_sample, pca_dim=50, perplexity=40, n_iter=1000, seed=42):
    n = len(X_sample)
    print(f"\nPCA {X_sample.shape[1]}→{pca_dim} on {n:,} sample points …")
    pca  = PCA(n_components=pca_dim, random_state=seed)
    X_pca = pca.fit_transform(X_sample)
    var   = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  Variance explained by {pca_dim} PCs: {var*100:.1f}%")

    perp = min(perplexity, (n - 1) // 3)
    print(f"t-SNE perplexity={perp} on {n:,} points …")
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter,
                random_state=seed, init="pca", learning_rate="auto",
                n_jobs=1)           # n_jobs=1 avoids Windows wmic crash
    return tsne.fit_transform(X_pca)


# ── 5. Plots ───────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def plot_by_benchmark(xy, df_s, blist):
    fig, ax = plt.subplots(figsize=(13, 9))
    colors = cm.tab20(np.linspace(0, 1, len(blist)))
    for col, bname in zip(colors, blist):
        mask = df_s["benchmark"].values == bname
        n = mask.sum()
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[col],
                   label=f"{bname} ({n:,})", s=8, alpha=0.55, linewidths=0)
    ax.legend(markerscale=2, fontsize=8, loc="best", ncol=2)
    ax.set_title("t-SNE (stratified sample): coloured by benchmark", fontsize=13)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(True, alpha=0.2)
    _save(fig, "fig1_by_benchmark.png")


def plot_by_label(xy, df_s, blist):
    binary = [b for b in blist if b not in TASK_TYPE and b in df_s["benchmark"].values]
    n = len(binary)
    if n == 0:
        print("  (no binary benchmarks to plot)"); return
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, bname in zip(axes, binary):
        mask    = df_s["benchmark"].values == bname
        sub_xy  = xy[mask]
        lbl     = df_s.loc[mask, "label_num"].values
        ax.scatter(sub_xy[lbl == 0, 0], sub_xy[lbl == 0, 1],
                   c="steelblue", s=7, alpha=0.5, label="No (0)")
        ax.scatter(sub_xy[lbl == 1, 0], sub_xy[lbl == 1, 1],
                   c="tomato",    s=7, alpha=0.7, label="Yes (1)")
        ax.set_title(bname, fontsize=10); ax.legend(fontsize=7, markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("t-SNE per benchmark: active (red) vs inactive (blue)", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig2_by_label.png")


def plot_kmeans_clusters(xy, cluster_s, n_clusters):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = cm.tab20(np.linspace(0, 1, n_clusters))
    for k in range(n_clusters):
        mask = cluster_s == k
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[colors[k]],
                   label=f"C{k} ({mask.sum():,})", s=8, alpha=0.6, linewidths=0)
    ax.legend(markerscale=2, fontsize=7, loc="best", ncol=4)
    ax.set_title(f"t-SNE: MiniBatchKMeans clusters (k={n_clusters}, sample)", fontsize=13)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(True, alpha=0.2)
    _save(fig, "fig3_kmeans_clusters.png")


def plot_label_distribution(df_full, blist):
    binary = [b for b in blist if b not in TASK_TYPE and b in df_full["benchmark"].values]
    counts = []
    for bname in binary:
        sub = df_full[df_full["benchmark"] == bname]["label_num"]
        pos, neg = (sub == 1).sum(), (sub == 0).sum()
        counts.append({"benchmark": bname, "positive": pos, "negative": neg,
                       "pos_ratio": pos / (pos + neg) if (pos + neg) > 0 else 0})
    cdf = pd.DataFrame(counts)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x, w = np.arange(len(cdf)), 0.35
    axes[0].bar(x - w/2, cdf["positive"], w, label="Positive (Yes)", color="tomato")
    axes[0].bar(x + w/2, cdf["negative"], w, label="Negative (No)",  color="steelblue")
    axes[0].set_xticks(x); axes[0].set_xticklabels(cdf["benchmark"], rotation=45, ha="right")
    axes[0].set_ylabel("Count (all molecules)"); axes[0].set_title("Label counts per benchmark")
    axes[0].legend()

    axes[1].bar(x, cdf["pos_ratio"], color="mediumseagreen")
    axes[1].axhline(0.5, ls="--", color="gray", alpha=0.7, label="50% balance")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cdf["benchmark"], rotation=45, ha="right")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Positive ratio")
    axes[1].set_title("Class imbalance: fraction of positives (all molecules)")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "fig4_label_distribution.png")


def plot_benchmark_similarity(X_full, df_full, blist):
    present = [b for b in blist if b in df_full["benchmark"].values]
    centers = np.array([X_full[df_full["benchmark"].values == b].mean(axis=0) for b in present])
    normed  = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9)
    sim     = normed @ normed.T

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Cosine similarity between benchmark mean SBERT embeddings\n(computed on ALL molecules)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig5_benchmark_similarity.png")


def plot_cluster_composition(df_full, cluster_full, blist):
    """Stacked bar using ALL molecule cluster assignments."""
    df2 = df_full.copy()
    df2["cluster"] = cluster_full
    n_clusters = int(cluster_full.max()) + 1
    present = [b for b in blist if b in df_full["benchmark"].values]
    colors  = cm.tab20(np.linspace(0, 1, len(present)))

    pivot     = df2.groupby(["cluster", "benchmark"]).size().unstack(fill_value=0)
    pivot     = pivot.reindex(columns=present, fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(max(12, n_clusters * 0.7), 6))
    bottom = np.zeros(len(pivot_pct))
    for col, color in zip(present, colors):
        vals = pivot_pct[col].values if col in pivot_pct.columns else np.zeros(len(pivot_pct))
        ax.bar(pivot_pct.index, vals, bottom=bottom, label=col, color=color)
        bottom += vals
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("% of molecules")
    ax.set_title("Benchmark composition per cluster (ALL molecules)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticks(range(n_clusters))
    fig.tight_layout()
    _save(fig, "fig6_cluster_composition.png")


def plot_by_task_label_unified(xy, df_s, blist):
    """
    Figure 7: single t-SNE, all benchmarks together, coloured by label.
    Active (Yes=1) = red, Inactive (No=0) = blue, Unknown/regression = grey.
    Each benchmark gets its own marker shape so you can read both dimensions.
    """
    binary = [b for b in blist if b not in TASK_TYPE and b in df_s["benchmark"].values]
    markers = ["o","s","^","D","v","P","X","*","h","p","H","8","<",">"]
    bmark2marker = {b: markers[i % len(markers)] for i, b in enumerate(blist)}

    fig, ax = plt.subplots(figsize=(14, 10))

    # grey background: regression / unknown label molecules
    mask_grey = ~df_s["benchmark"].isin(binary) | df_s["label_num"].isna()
    ax.scatter(xy[mask_grey, 0], xy[mask_grey, 1],
               c="lightgrey", s=6, alpha=0.3, linewidths=0, label="unknown/regression", zorder=1)

    for bname in binary:
        m = bmark2marker[bname]
        mask_b = df_s["benchmark"].values == bname
        lbl    = df_s.loc[mask_b, "label_num"].values

        pos = mask_b & (df_s["label_num"].values == 1)
        neg = mask_b & (df_s["label_num"].values == 0)

        if pos.sum():
            ax.scatter(xy[pos, 0], xy[pos, 1], c="tomato", marker=m,
                       s=14, alpha=0.75, linewidths=0,
                       label=f"{bname} active ({pos.sum():,})", zorder=3)
        if neg.sum():
            ax.scatter(xy[neg, 0], xy[neg, 1], c="steelblue", marker=m,
                       s=10, alpha=0.45, linewidths=0,
                       label=f"{bname} inactive ({neg.sum():,})", zorder=2)

    ax.legend(markerscale=1.5, fontsize=7, loc="best", ncol=2,
              framealpha=0.8, title="benchmark + activity")
    ax.set_title("t-SNE: all benchmarks unified — red=active, blue=inactive\n"
                 "(marker shape = benchmark)", fontsize=13)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(True, alpha=0.15)
    _save(fig, "fig7_unified_task_label.png")


def plot_task_label_subplots_shared_axes(xy, df_s, blist):
    """
    Figure 8: one subplot per binary benchmark, all sharing the same t-SNE
    coordinate space. Grey = other benchmarks (context), red/blue = this
    benchmark's labels. Lets you see each task's active region in full context.
    """
    binary = [b for b in blist if b not in TASK_TYPE and b in df_s["benchmark"].values]
    n = len(binary)
    if n == 0:
        print("  (no binary benchmarks — skipping fig8)"); return

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for ax, bname in zip(axes, binary):
        # grey background: all OTHER molecules for spatial context
        other = df_s["benchmark"].values != bname
        ax.scatter(xy[other, 0], xy[other, 1],
                   c="lightgrey", s=4, alpha=0.2, linewidths=0, zorder=1)

        mask = df_s["benchmark"].values == bname
        lbl  = df_s.loc[mask, "label_num"].values
        pos  = lbl == 1
        neg  = lbl == 0
        n_pos, n_neg = pos.sum(), neg.sum()

        ax.scatter(xy[mask][neg, 0], xy[mask][neg, 1],
                   c="steelblue", s=8, alpha=0.5, linewidths=0,
                   label=f"inactive ({n_neg:,})", zorder=2)
        ax.scatter(xy[mask][pos, 0], xy[mask][pos, 1],
                   c="tomato", s=10, alpha=0.8, linewidths=0,
                   label=f"active ({n_pos:,})", zorder=3)

        ratio = n_pos / (n_pos + n_neg) * 100 if (n_pos + n_neg) > 0 else 0
        ax.set_title(f"{bname}\n{ratio:.1f}% active", fontsize=9)
        ax.legend(fontsize=7, markerscale=1.5, loc="lower right")
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("t-SNE per task: active (red) vs inactive (blue) in full chemical context (grey)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig8_task_label_in_context.png")


def smiles_features(smiles):
    """
    Extract simple structural features from a SMILES string without RDKit.
    All features are counts or boolean flags readable directly from SMILES notation.
    """
    import re
    s = str(smiles)
    # aromatic atoms: lowercase c n o s p b
    n_arom   = len(re.findall(r'[cnospb]', s))
    # ring closure digits (rough ring count — each ring opens and closes once)
    ring_dig  = re.findall(r'(?<![%])\d|%\d{2}', s)
    n_rings   = len(ring_dig) // 2
    # heavy atom count: uppercase + lowercase element letters, skip H
    n_heavy   = len(re.findall(r'[A-Z][a-z]?|[cnospb]', s))
    # heteroatoms
    has_N     = int(bool(re.search(r'[Nn]', s)))
    has_O     = int(bool(re.search(r'[Oo]', s)))
    has_S     = int(bool(re.search(r'[Ss]', s)))
    has_P     = int(bool(re.search(r'[Pp]', s)))
    has_hal   = int(bool(re.search(r'F|Cl|Br|I', s)))
    has_F     = int('F' in s)
    has_Cl    = int('Cl' in s)
    has_Br    = int('Br' in s)
    # charge / radical
    has_charge = int(bool(re.search(r'[+\-]', s)))
    # branches: each '(' opens a branch
    n_branches = s.count('(')
    # double / triple bonds
    n_double   = s.count('=')
    n_triple   = s.count('#')
    # aromatic fraction
    arom_frac  = n_arom / max(n_heavy, 1)
    return {
        "n_heavy": n_heavy, "n_rings": n_rings, "n_arom": n_arom,
        "arom_frac": arom_frac, "n_branches": n_branches,
        "n_double": n_double, "n_triple": n_triple,
        "has_N": has_N, "has_O": has_O, "has_S": has_S,
        "has_P": has_P, "has_hal": has_hal,
        "has_F": has_F, "has_Cl": has_Cl, "has_Br": has_Br,
        "has_charge": has_charge,
    }


def plot_cluster_task_activity(df_full, cluster_full, blist):
    """
    Figure 9: heatmap — rows = clusters, cols = binary benchmarks.
    Cell value = fraction of that benchmark's molecules in that cluster that are ACTIVE.
    Reveals which clusters are 'hot zones' for which tasks.
    Also prints a ranked table: for each task, top-3 clusters by activity rate.
    """
    binary = [b for b in blist if b not in TASK_TYPE and b in df_full["benchmark"].values]
    if not binary:
        print("  (no binary benchmarks — skipping fig9)"); return

    df2 = df_full.copy()
    df2["cluster"] = cluster_full
    n_clusters = int(cluster_full.max()) + 1

    # build matrix: rows=clusters, cols=benchmarks → mean activity rate
    mat = np.full((n_clusters, len(binary)), np.nan)
    mat_n = np.zeros((n_clusters, len(binary)), dtype=int)   # sample size per cell

    for j, bname in enumerate(binary):
        sub = df2[df2["benchmark"] == bname]
        for k in range(n_clusters):
            cell = sub[sub["cluster"] == k]["label_num"].dropna()
            if len(cell) >= 3:     # only show cells with enough data
                mat[k, j]   = cell.mean()
                mat_n[k, j] = len(cell)

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(binary) * 1.2), max(8, n_clusters * 0.55)),
                             gridspec_kw={"width_ratios": [3, 1]})

    # left: activity rate heatmap
    ax = axes[0]
    masked = np.ma.array(mat, mask=np.isnan(mat))
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("whitesmoke")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(binary))); ax.set_xticklabels(binary, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_clusters)); ax.set_yticklabels([f"C{k}" for k in range(n_clusters)], fontsize=8)
    ax.set_xlabel("Benchmark (task)"); ax.set_ylabel("Cluster")
    ax.set_title("Active fraction per cluster × task\n(white = < 3 molecules)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="fraction active")

    # annotate cells with n
    for k in range(n_clusters):
        for j in range(len(binary)):
            if not np.isnan(mat[k, j]):
                ax.text(j, k, f"{mat[k,j]:.2f}\nn={mat_n[k,j]}",
                        ha="center", va="center", fontsize=6,
                        color="black" if 0.2 < mat[k, j] < 0.8 else "white")

    # right: cluster size bar
    ax2 = axes[1]
    sizes = pd.Series(cluster_full).value_counts().sort_index()
    ax2.barh(sizes.index, sizes.values, color="steelblue", alpha=0.7)
    ax2.set_yticks(range(n_clusters)); ax2.set_yticklabels([f"C{k}" for k in range(n_clusters)], fontsize=8)
    ax2.set_xlabel("Molecules in cluster"); ax2.set_title("Cluster size", fontsize=10)
    ax2.invert_yaxis()

    fig.suptitle("Which clusters are 'hot zones' for which tasks?", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig9_cluster_task_activity.png")

    # print ranked table
    print("\n  Top-3 clusters by activity rate per task:")
    for j, bname in enumerate(binary):
        col = mat[:, j]
        valid = [(k, col[k]) for k in range(n_clusters) if not np.isnan(col[k])]
        top3 = sorted(valid, key=lambda x: -x[1])[:3]
        row  = "  ".join([f"C{k}={v:.2f}(n={mat_n[k,j]})" for k, v in top3])
        print(f"    {bname:<25} → {row}")


def plot_smiles_features_per_cluster(df_full, cluster_full, blist):
    """
    Figure 10: structural fingerprint per cluster from SMILES string parsing.
    Shows mean value of each feature per cluster as a heatmap, so you can read
    'cluster 3 = large aromatic molecules with halogens'.
    Also overlays a t-SNE-independent ranked table per cluster.
    """
    print("  Extracting SMILES features …")
    feat_rows = [smiles_features(s) for s in df_full["graph"].tolist()]
    feat_df   = pd.DataFrame(feat_rows)
    feat_df["cluster"] = cluster_full

    FEAT_LABELS = {
        "n_heavy":    "Heavy atoms",  "n_rings":   "Ring closures",
        "n_arom":     "Aromatic atoms","arom_frac": "Aromatic fraction",
        "n_branches": "Branches ()",  "n_double":  "Double bonds (=)",
        "n_triple":   "Triple bonds (#)","has_N":  "Has N",
        "has_O":      "Has O",        "has_S":    "Has S",
        "has_P":      "Has P",        "has_hal":  "Has halogen",
        "has_F":      "Has F",        "has_Cl":   "Has Cl",
        "has_Br":     "Has Br",       "has_charge":"Has charge",
    }
    feats = list(FEAT_LABELS.keys())

    # mean per cluster, z-score across clusters for relative comparison
    cluster_means = feat_df.groupby("cluster")[feats].mean()
    z = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-9)

    n_clusters = int(cluster_full.max()) + 1
    fig, axes = plt.subplots(1, 2, figsize=(22, max(8, n_clusters * 0.55)),
                             gridspec_kw={"width_ratios": [2, 1]})

    # left: z-score heatmap (relative — which clusters stand out on each feature)
    ax = axes[0]
    im = ax.imshow(z.values, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([FEAT_LABELS[f] for f in feats], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(n_clusters)); ax.set_yticklabels([f"C{k}" for k in range(n_clusters)], fontsize=8)
    ax.set_title("SMILES structural features per cluster (z-score)\nBlue=below avg  Red=above avg", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="z-score")

    # right: raw mean for top-3 discriminating features per cluster
    ax2 = axes[1]
    ax2.axis("off")
    top_feat = z.abs().mean(axis=0).nlargest(5).index.tolist()
    lines = ["Top-5 most discriminating features:\n"]
    for f in top_feat:
        lines.append(f"  {FEAT_LABELS[f]}")
    lines.append("\nCluster structural summary (top feature):")
    for k in range(n_clusters):
        row    = z.loc[k] if k in z.index else pd.Series(dtype=float)
        if len(row):
            top_f  = row.abs().idxmax()
            direct = "↑ high" if row[top_f] > 0 else "↓ low"
            lines.append(f"  C{k}: {direct} {FEAT_LABELS.get(top_f, top_f)}")
    ax2.text(0.02, 0.98, "\n".join(lines), transform=ax2.transAxes,
             fontsize=8, va="top", family="monospace",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    fig.suptitle("What chemistry characterises each cluster? (SMILES-derived features)", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig10_smiles_features_per_cluster.png")

    # save readable CSV
    out_csv = os.path.join(OUT_DIR, "cluster_smiles_features.csv")
    cluster_means.round(3).to_csv(out_csv)
    print(f"  saved mean features table → {out_csv}")


def plot_label_disagreement(df_full, X_full, cluster_full, blist):
    """
    Figures 11 + 12: why do structurally similar molecules have different labels?

    Fig 11 — Cross-task label agreement matrix.
      For every pair of binary benchmarks, find molecules that appear in BOTH
      (matched by identical SMILES string), then compute:
        - agreement rate  (both active or both inactive)
        - conflict rate   (active in one, inactive in the other)
      High conflict = same structure, opposite function → different biological target.

    Fig 12 — Per-cluster label entropy across tasks.
      Within each cluster, compute the Shannon entropy of the label distribution
      per benchmark. High entropy = cluster contains both actives and inactives
      for a task → structure alone doesn't determine label for that task.
      Plotted on the t-SNE sample so you can see WHERE in chemical space
      label ambiguity is highest.
    """
    from scipy.stats import entropy as sp_entropy

    binary = [b for b in blist if b not in TASK_TYPE and b in df_full["benchmark"].values]
    if len(binary) < 2:
        print("  (need ≥2 binary benchmarks for disagreement analysis — skipping)"); return

    # ── Fig 11: cross-task label agreement ──────────────────────────────────
    print("  Computing cross-task label agreement …")
    n_b = len(binary)
    agree_mat   = np.full((n_b, n_b), np.nan)
    conflict_mat = np.full((n_b, n_b), np.nan)
    n_shared_mat = np.zeros((n_b, n_b), dtype=int)

    # build smiles→label lookup per benchmark
    smiles2label = {}
    for bname in binary:
        sub = df_full[df_full["benchmark"] == bname][["graph", "label_num"]].dropna()
        smiles2label[bname] = dict(zip(sub["graph"], sub["label_num"]))

    for i, b1 in enumerate(binary):
        for j, b2 in enumerate(binary):
            if i >= j:
                continue
            shared = set(smiles2label[b1]) & set(smiles2label[b2])
            n_shared_mat[i, j] = n_shared_mat[j, i] = len(shared)
            if len(shared) < 5:
                continue
            agree = conflict = 0
            for smi in shared:
                l1, l2 = smiles2label[b1][smi], smiles2label[b2][smi]
                if l1 == l2:   agree   += 1
                else:          conflict += 1
            total = agree + conflict
            if total:
                agree_mat[i, j]    = agree_mat[j, i]    = agree   / total
                conflict_mat[i, j] = conflict_mat[j, i] = conflict / total

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, mat, title, cmap in [
        (axes[0], agree_mat,    "Label AGREEMENT rate\n(same label for same SMILES)", "YlGn"),
        (axes[1], conflict_mat, "Label CONFLICT rate\n(opposite label for same SMILES)", "YlOrRd"),
    ]:
        masked = np.ma.array(mat, mask=np.isnan(mat))
        cm_use = plt.cm.get_cmap(cmap)
        cm_use.set_bad("whitesmoke")
        im = ax.imshow(masked, cmap=cm_use, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(n_b)); ax.set_xticklabels(binary, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_b)); ax.set_yticklabels(binary, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(n_b):
            for j in range(n_b):
                if not np.isnan(mat[i, j]):
                    txt = f"{mat[i,j]:.2f}\n(n={n_shared_mat[i,j]})"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                            color="black" if mat[i, j] < 0.85 else "white")
        ax.set_title(title, fontsize=11)

    fig.suptitle(
        "Same SMILES, different task → different label?\n"
        "White = fewer than 5 shared molecules between that pair",
        fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig11_cross_task_label_conflict.png")

    # print highest-conflict pairs
    print("\n  Highest label-conflict pairs (same molecule, opposite label):")
    pairs = []
    for i in range(n_b):
        for j in range(i+1, n_b):
            if not np.isnan(conflict_mat[i, j]) and n_shared_mat[i, j] >= 10:
                pairs.append((binary[i], binary[j], conflict_mat[i, j], n_shared_mat[i, j]))
    for b1, b2, c, n in sorted(pairs, key=lambda x: -x[2])[:8]:
        print(f"    {b1:<20} vs {b2:<20}  conflict={c:.2f}  shared_n={n}")

    # ── Fig 12: per-cluster label entropy on t-SNE ───────────────────────────
    # compute mean label entropy per cluster across all binary benchmarks
    df2 = df_full.copy()
    df2["cluster"] = cluster_full
    n_clusters = int(cluster_full.max()) + 1

    cluster_entropy = np.zeros(n_clusters)
    for k in range(n_clusters):
        sub  = df2[df2["cluster"] == k]
        ents = []
        for bname in binary:
            lvals = sub[sub["benchmark"] == bname]["label_num"].dropna().values
            if len(lvals) < 5: continue
            p1 = lvals.mean()
            p0 = 1 - p1
            if 0 < p1 < 1:
                ents.append(sp_entropy([p0, p1], base=2))   # max = 1.0 bit
        cluster_entropy[k] = np.mean(ents) if ents else 0.0

    # map cluster entropy back to t-SNE sample points
    # (we need the full df + cluster_full to get sample indices)
    # use the sample df index passed via df_s — but we only have df_full here.
    # Instead, build a cluster→entropy lookup and expose it via return value.
    print(f"\n  Cluster label entropy (0=pure, 1=maximum disagreement):")
    for k in range(n_clusters):
        bar = "█" * int(cluster_entropy[k] * 20)
        print(f"    C{k:2d}: {cluster_entropy[k]:.3f}  {bar}")

    return cluster_entropy   # caller uses this for fig 12


def plot_label_entropy_on_tsne(xy, df_s, cluster_s, cluster_entropy):
    """
    Figure 12: t-SNE scatter where each point is coloured by its cluster's
    mean label entropy. Warm = high disagreement (structure doesn't predict label).
    Cool = low entropy (structure strongly predicts label for most tasks).
    """
    if cluster_entropy is None:
        return
    point_entropy = np.array([cluster_entropy[k] for k in cluster_s])

    fig, ax = plt.subplots(figsize=(13, 9))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=point_entropy,
                    cmap="RdYlBu_r", s=8, alpha=0.7, linewidths=0,
                    vmin=0, vmax=1)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Mean label entropy across tasks\n(0 = structure predicts label, 1 = complete disagreement)")
    ax.set_title(
        "Where does structure FAIL to predict task labels?\n"
        "Warm regions = same chemical scaffold, opposite activity across benchmarks",
        fontsize=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(True, alpha=0.15)
    _save(fig, "fig12_label_entropy_on_tsne.png")


def plot_interactive(xy, df_s, cluster_s):
    df2 = df_s.copy()
    df2["x"] = xy[:, 0]; df2["y"] = xy[:, 1]
    df2["cluster"] = cluster_s.astype(str)
    df2["smiles_short"] = df2["graph"].apply(
        lambda s: (str(s)[:60] + "…") if len(str(s)) > 60 else str(s))
    df2["activity"] = df2["label_num"].map(
        {1.0: "active", 0.0: "inactive"}).fillna("unknown")

    common_hover = {"smiles_short": True, "label": True, "activity": True,
                    "split": True, "cluster": True, "benchmark": True,
                    "x": False, "y": False}

    # colour by benchmark
    fig1 = px.scatter(
        df2, x="x", y="y", color="benchmark", symbol="cluster",
        hover_data=common_hover,
        title=f"Interactive t-SNE — colour: benchmark  ({len(df2):,} sample)",
        width=1300, height=850, opacity=0.65,
    )
    fig1.update_traces(marker=dict(size=5))
    p1 = os.path.join(OUT_DIR, "interactive_by_benchmark.html")
    fig1.write_html(p1); print(f"  saved → {p1}")

    # colour by activity
    color_map = {"active": "tomato", "inactive": "steelblue", "unknown": "lightgrey"}
    fig2 = px.scatter(
        df2, x="x", y="y", color="activity", symbol="benchmark",
        color_discrete_map=color_map,
        hover_data=common_hover,
        title=f"Interactive t-SNE — colour: task label  ({len(df2):,} sample)",
        width=1300, height=850, opacity=0.7,
    )
    fig2.update_traces(marker=dict(size=5))
    p2 = os.path.join(OUT_DIR, "interactive_by_task_label.html")
    fig2.write_html(p2); print(f"  saved → {p2}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks",    nargs="+", default=ALL_BENCHMARKS)
    parser.add_argument("--max_per_split", type=int,  default=0,
                        help="Cap molecules per benchmark (0 = all)")
    parser.add_argument("--n_clusters",    type=int,  default=13)
    parser.add_argument("--tsne_sample",   type=int,  default=15000,
                        help="Points to use for t-SNE (stratified sample from full data)")
    parser.add_argument("--pca_dim",       type=int,  default=50,
                        help="PCA dims before t-SNE (reduces noise, speeds up t-SNE)")
    parser.add_argument("--perplexity",    type=float, default=40)
    parser.add_argument("--tsne_iter",     type=int,  default=1000)
    parser.add_argument("--batch_size",    type=int,  default=256)
    parser.add_argument("--no_cache",      action="store_true",
                        help="Re-compute SBERT embeddings even if cache exists")
    args = parser.parse_args()

    # ── step 1: load ──────────────────────────────────────────────────────────
    print("\n=== Step 1: Loading benchmarks ===")
    df = load_benchmarks(args.benchmarks, args.max_per_split)
    print(f"\nTotal molecules: {len(df):,}  |  Benchmarks: {df['benchmark'].nunique()}")
    print(df.groupby("benchmark").size().rename("count").to_string())

    smiles_list = df["graph"].tolist()

    # ── step 2: encode ────────────────────────────────────────────────────────
    print("\n=== Step 2: SBERT encoding ===")
    if os.path.exists(EMBED_CACHE) and not args.no_cache:
        npz   = np.load(EMBED_CACHE)
        X_raw = npz["X"]
        print(f"  Loaded cache: shape={X_raw.shape}")
        if len(X_raw) != len(df):
            print(f"  Cache has {len(X_raw):,} rows but data has {len(df):,} — re-encoding …")
            X_raw = encode_smiles(smiles_list, batch_size=args.batch_size)
            np.savez(EMBED_CACHE, X=X_raw)
    else:
        X_raw = encode_smiles(smiles_list, batch_size=args.batch_size)
        np.savez(EMBED_CACHE, X=X_raw)
        print(f"  Saved to {EMBED_CACHE}")

    # L2-normalise
    X = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-9)

    # ── step 3: cluster on ALL data ───────────────────────────────────────────
    print("\n=== Step 3: Clustering (ALL molecules) ===")
    cluster_full, km = run_kmeans(X, n_clusters=args.n_clusters)

    # ── step 4: stratified sample → PCA → t-SNE ───────────────────────────────
    print(f"\n=== Step 4: Stratified t-SNE sample (target {args.tsne_sample:,} points) ===")
    X_s, df_s, cluster_s, _ = build_tsne_sample(
        X, df, cluster_full, args.tsne_sample)
    print(f"  Sample size: {len(df_s):,} molecules")
    xy = run_pca_tsne(X_s, pca_dim=args.pca_dim,
                      perplexity=args.perplexity, n_iter=args.tsne_iter)

    # ── step 5: plots ─────────────────────────────────────────────────────────
    print("\n=== Step 5: Generating plots ===")
    blist = [b for b in args.benchmarks if b in df["benchmark"].values]

    plot_by_benchmark(xy, df_s, blist)
    plot_by_label(xy, df_s, blist)
    plot_kmeans_clusters(xy, cluster_s, args.n_clusters)
    plot_label_distribution(df, blist)              # full data
    plot_benchmark_similarity(X_raw, df, blist)     # full data
    plot_cluster_composition(df, cluster_full, blist)  # full data
    plot_by_task_label_unified(xy, df_s, blist)
    plot_task_label_subplots_shared_axes(xy, df_s, blist)
    plot_cluster_task_activity(df, cluster_full, blist)
    plot_smiles_features_per_cluster(df, cluster_full, blist)
    cluster_entropy = plot_label_disagreement(df, X, cluster_full, blist)
    plot_label_entropy_on_tsne(xy, df_s, cluster_s, cluster_entropy)
    plot_interactive(xy, df_s, cluster_s)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    df["cluster"] = cluster_full
    print("\nCluster sizes (ALL molecules):")
    print(df["cluster"].value_counts().sort_index().rename("n_molecules").to_string())
    print("\nMolecules per benchmark × split:")
    print(df.groupby(["benchmark", "split"]).size().to_string())
    print(f"\nAll figures → {OUT_DIR}")


if __name__ == "__main__":
    main()
