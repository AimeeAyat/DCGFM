# Experiment Results

All experiments use: **30 epochs**, **batch 256**, **LLM = SentenceTransformer (ST, 768d)**, **hard pruning epochs = 25**, **seed = 1**, **test_rep = 10**.

Pruning direction:
- **Near-center** (`hard_pruning_reverse: true`) — removes redundant/common graphs, keeps diverse ones
- **Far-center** (`hard_pruning_reverse: false`) — removes outlier/anomalous graphs, keeps representative ones

---

## Baseline Experiments (All Datasets: arxiv + FB15K237 + mol)

### 1. Hard 70% Only — Near-center pruning
**Dir:** `2026-04-15_10-11-10 (hard prune 70 only)`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, 5 layers |
| Hard pruning | 70%, GIN-SVDD, near-center |
| Soft pruning | None (prune_ratio=0) |

Results: not recorded in command file.

---

### 2. Hard 70% + Soft 50% — Near-center pruning
**Dir:** `2026-04-15_21-22-57 (hard 70, soft 50)`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, 5 layers |
| Hard pruning | 70%, GIN-SVDD, near-center |
| Soft pruning | 50% (InfoBatch) |

Results: not recorded in command file.

---

### 3. GraphGPS Hard 70% + Soft 50% — Near-center pruning
**Dir:** `2026-04-16_01-07-16 (GraphGPS hard 70, soft 50)`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, 5 layers |
| Hard pruning | 70%, **GraphGPS-SVDD**, near-center |
| Soft pruning | 50% (InfoBatch) |

Results: not recorded in command file.

---

### 4. Hard 70% + Soft 50% — Partial test
**Dir:** `2026-04-16_11-01-25 (testing only hard 70, soft 50)`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, 5 layers |
| Hard pruning | 70%, GIN-SVDD, near-center |
| Soft pruning | 50% (InfoBatch) |

| Metric | Score |
|---|---|
| test_arxiv_fs_50 / acc | 0.4261 ± 0.0116 |
| test_pubmed_fs_30 / acc | 0.3054 ± 0.0049 |
| test_FB15K237_fs_200 / acc | 0.3508 ± 0.0087 |
| test_WN18RR_fs_100 / acc | 0.1551 ± 0.0044 |
| test_chemhiv_fs_20 / auc | 0.5354 ± 0.0066 |
| test_chempcba_fs_20 / aucmulti | 0.5275 ± 0.0114 |

---

### 5. ★ FULL BASELINE — Hard 70% + Soft 30% — Near-center pruning
**Dir:** `2026-04-17_17-40-10 (results (hard 70, soft 30))`

**This is the primary baseline** used for comparison with all subsequent experiments.

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, 5 layers |
| Hard pruning | 70%, GIN-SVDD, **near-center** |
| Soft pruning | 30% (InfoBatch) |
| Dropout | 0.0 |

| Metric | Score |
|---|---|
| test_arxiv_fs_50 / acc | **0.4591 ± 0.0088** |
| test_arxiv_fs_30 / acc | **0.6059 ± 0.0068** |
| test_cora_fs_20 / acc | **0.5242 ± 0.0134** |
| test_cora_fs_70 / acc | 0.1517 ± 0.0067 |
| test_pubmed_fs_30 / acc | 0.3114 ± 0.0133 |
| test_wikics_fs_100 / acc | 0.1028 ± 0.0060 |
| test_FB15K237_fs_100 / acc | **0.5174 ± 0.0134** |
| test_FB15K237_fs_200 / acc | 0.3684 ± 0.0100 |
| test_WN18RR_fs_50 / acc | 0.2225 ± 0.0052 |
| test_WN18RR_fs_100 / acc | 0.1097 ± 0.0063 |
| test_chemhiv_fs_20 / auc | 0.4825 ± 0.0169 |
| test_chempcba_fs_20 / aucmulti | **0.5421 ± 0.0092** |

---

## Molecular Architecture Experiments (mol_fs only)

All experiments below use **far-center pruning** (`hard_pruning_reverse: false`), hard 70%, soft 30%, dropout 0.1.

> **Motivation:** Hop distribution analysis showed 81% of chemblpre molecules have diameter > 10. Standard 5-layer GIN covers only 1.4% of molecules fully.

---

### 6. Far-center pruning — Default GIN (mol only)
**Dir:** `2026-05-16_11-17-21 pruning fathest graphs`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, 5 layers |
| Hard pruning | 70%, **far-center** |
| Soft pruning | 30% |

| Metric | Score |
|---|---|
| test_chemhiv_fs_20 / auc | 0.5360 ± 0.0153 |
| test_chempcba_fs_20 / aucmulti | 0.5146 ± 0.0063 |

---

### 7. Deep GIN — 10 layers + Residuals (mol only)
**Dir:** `2026-05-16_14-40-08 10 gin layers`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, **10 layers + residual connections** |
| Hard pruning | 70%, far-center |
| Soft pruning | 30% |

| Metric | Score |
|---|---|
| test_chemhiv_fs_20 / auc | **0.5757 ± 0.0105** |
| test_chempcba_fs_20 / aucmulti | 0.4971 ± 0.0076 |

---

### 8. Deep GIN — 15 layers + Residuals (mol only)
**Dir:** `2026-05-16_16-25-59 15 gin layers`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, **15 layers + residual connections** |
| Hard pruning | 70%, far-center |
| Soft pruning | 30% |

| Metric | Score |
|---|---|
| test_chemhiv_fs_20 / auc | 0.4853 ± 0.0079 |
| test_chempcba_fs_20 / aucmulti | 0.5167 ± 0.0079 |

> 15 layers underperforms 10 layers — over-smoothing despite residuals.

---

### 9. Adaptive-depth GIN — 20 layers + Diameter-gated Attention (mol only)
**Dir:** `2026-05-16_17-50-22 gin adaptive`

| Setting | Value |
|---|---|
| GNN | Single 20-layer residual GIN; nodes attend only to layers ≤ diameter bucket (≤5 → layers 1-5, 6-10 → layers 1-10, >10 → all 20) |
| Hard pruning | 70%, far-center |
| Soft pruning | 30% |

| Metric | Score |
|---|---|
| test_chemhiv_fs_20 / auc | 0.5525 ± 0.0146 |
| test_chempcba_fs_20 / aucmulti | 0.5200 ± 0.0105 |

---

### 10. Virtual Node GIN — 10 layers (mol only)
**Dir:** `2026-05-16_19-37-17 gin 10-vn mol only`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, 10 layers + **corrected virtual node** |
| Hard pruning | 70%, far-center |
| Soft pruning | 30% |

> Virtual node: a global node connected to all atoms, updated each layer. Shortcircuits long-range dependencies.

| Metric | Score |
|---|---|
| test_chemhiv_fs_20 / auc | 0.5581 ± 0.0098 |
| test_chempcba_fs_20 / aucmulti | **0.5409 ± 0.0148** |

---

## Full Multi-dataset Experiments with VN

### 11. VN GIN 5 layers — All Datasets
**Dir:** `2026-05-16_20-08-55 vn gin 5 all dataset`

| Setting | Value |
|---|---|
| GNN | RGCN-Edge, 5 layers + virtual node |
| Hard pruning | 70%, far-center |
| Soft pruning | 30% |

| Metric | Score |
|---|---|
| test_arxiv_fs_50 / acc | 0.3662 ± 0.0111 |
| test_arxiv_fs_30 / acc | 0.5093 ± 0.0091 |
| test_cora_fs_20 / acc | 0.5033 ± 0.0136 |
| test_cora_fs_70 / acc | 0.1503 ± 0.0063 |
| test_pubmed_fs_30 / acc | 0.3170 ± 0.0103 |
| test_wikics_fs_100 / acc | 0.0724 ± 0.0041 |
| test_FB15K237_fs_100 / acc | 0.5330 ± 0.0121 |
| test_FB15K237_fs_200 / acc | 0.3900 ± 0.0087 |
| test_WN18RR_fs_50 / acc | 0.2991 ± 0.0102 |
| test_WN18RR_fs_100 / acc | 0.2011 ± 0.0103 |
| test_chemhiv_fs_20 / auc | 0.5211 ± 0.0145 |

---

### 12. VN GIN 10 layers — All Datasets (In Progress)
**Dirs:** `2026-05-17_09-04-16`, `2026-05-17_17-26-50`, `2026-05-17_17-28-36`

Results pending.

---

## Summary Table — Molecular Results

| Experiment | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| Baseline (5L, near-center) | 0.4825 | **0.5421** |
| Far-center pruning (5L) | 0.5360 | 0.5146 |
| Deep GIN 10L + residuals | **0.5757** | 0.4971 |
| Deep GIN 15L + residuals | 0.4853 | 0.5167 |
| Adaptive-depth 20L | 0.5525 | 0.5200 |
| VN 10L (mol only) | 0.5581 | 0.5409 |
| VN 5L (all datasets) | 0.5211 | — |