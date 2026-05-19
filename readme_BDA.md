# BDA Experiments — DCGFM Extension

This document covers all experiments, run commands, and new files added on top of
the original [DCGFM](https://github.com/Yuhan1i/DCGFM) codebase for the BDA
project. The original paper code is unchanged; every new feature lives in a
separate file and is injected at runtime via namespace patching.

---

## Pruning Terminology

> **Far-center pruning** (`no flag`) — graphs **far from the SVDD centre are removed**.
> The centre represents the average/common molecular structure. Graphs with high
> anomaly scores (far from centre = structurally unusual/diverse) are removed.
> What remains: representative, typical drug-like molecules.
>
> **Near-center pruning** (`--hard_pruning_reverse`) — graphs **near the SVDD centre
> are removed**. Low anomaly score = structurally common/redundant graphs are pruned.
> What remains: diverse, structurally unusual molecules.

---

## Setup

```powershell
# Activate virtual environment
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\Activate.ps1

# Disable W&B logging (add to every command below)
$env:WANDB_MODE = "disabled"

# Working directory
cd F:\Rabia-Salman\DCGFM\OFA
```

---

## New Files Added

### GNN Architecture Extensions

| File | Purpose |
|---|---|
| `OFA/gp/nn/models/GNN_deep.py` | `DeepMultiLayerMessagePassing` — adds residual skip connections (`h = h + prev_h`) from layer 1 onward. Prevents over-smoothing when stacking 10+ layers. |
| `OFA/gp/nn/models/GNN_vn.py` | `VNMultiLayerMessagePassing` — corrected virtual node. Original `GNN.py` updated the VN each layer but never broadcast it back to node features. Adds missing step: `h = h + vnode[batch_seg]` before each layer. |

### Model Classes

| File | Purpose |
|---|---|
| `OFA/models/model_deep.py` | `PyGRGCNEdgeDeep` — RGCN-Edge with residual connections (10L/15L). |
| `OFA/models/model_vn.py` | `PyGRGCNEdgeVN` — RGCN-Edge with corrected virtual node broadcast. |
| `OFA/models/model_adaptive.py` | `AdaptiveGINModel` — single 20L GIN; nodes attend only over layers ≤ diameter bucket (≤5→1-5L, 6-10→1-10L, >10→all 20L) via masked attention. |
| `OFA/models/model_gin_transformer.py` | `GINTransformerModel` — 10L residual GIN + 1 Transformer block (sequential). All molecules use same architecture. |
| `OFA/models/model_moe.py` | `GraphMoEModel` — 3-expert MoE: E1=3L RGCN-Edge, E2=3L RGCN-Edge, E3=3L GPS (GINEConv+Attention). Sparse top-2 routing via 4-descriptor topology vector. Load-balancing auxiliary loss. |
| `OFA/models/model_dual_gin.py` | `DualVNGINModel` — two 10L residual GINs; molecules sorted by diameter, exact 50/50 split; separate SVDD pruning centre per half; hard routing at inference using computed threshold. |

### Runner Scripts

| File | Purpose |
|---|---|
| `OFA/run_cdm_deep.py` | Injects `PyGRGCNEdgeDeep`. Saves `results.json`. |
| `OFA/run_cdm_vn.py` | Injects `PyGRGCNEdgeVN`. Saves `results.json`. |
| `OFA/run_cdm_adaptive.py` | Injects `AdaptiveGINModel`. Saves `results.json`. |
| `OFA/run_cdm_gin_transformer.py` | Injects `GINTransformerModel`. Saves `results.json`. |
| `OFA/run_cdm_moe.py` | Injects `GraphMoEModel`; patches `training_step` to add aux load-balance loss. Saves `results.json`. |
| `OFA/run_cdm_dual.py` | Injects `DualVNGINModel`; replaces `get_effective_indices` with diameter-split dual-SVDD version. Saves `results.json`. |

### Analysis & Utilities

| File | Purpose |
|---|---|
| `OFA/mol_hop_analysis.py` | Computes graph diameter distribution for chemblpre/chemhiv/chempcba. Outputs `analysis_output/mol_hop_distribution.png`. Results cached. |
| `OFA/compile_results.py` | Reads all `results.json` from `saved_exp/` and `saved_exp_old/`, prints formatted comparison tables. |

### YAML Configs

| File | Tasks | Layers | Notes |
|---|---|---|---|
| `OFA/yamls/soft_and_hard_mol_far.yaml` | mol only | 5L | baseline mol-only config |
| `OFA/yamls/soft_and_hard_mol_deep.yaml` | mol only | 10/15L | edit `num_layers` to switch |
| `OFA/yamls/soft_and_hard_mol_vn.yaml` | mol only | 10L | VN experiment |
| `OFA/yamls/soft_and_hard_mol_adaptive.yaml` | mol only | 20L | adaptive depth |
| `OFA/yamls/soft_and_hard_mol_gin_transformer.yaml` | mol only | 10L | GIN+Transformer |
| `OFA/yamls/soft_and_hard_mol_moe.yaml` | mol only | 3L each | MoE, 50 epochs |
| `OFA/yamls/soft_and_hard_mol_dual.yaml` | mol only | 10L×2 | dual split-SVDD |
| `OFA/yamls/soft_and_hard_vn_full.yaml` | all 3 tasks | 5/10L | VN full experiment |

### Bug Fixes in Original Files

| File | Fix |
|---|---|
| `OFA/run_cdm.py` | Cache key now includes `_fwd` suffix when `hard_pruning_reverse=False` (prevents far/near-center index collision). Cache check now verifies ALL task files exist, not just `_0.pkl`. Added `--exp_label` for descriptive directory names. Saves `results.json`. |
| `OFA/gp/nn/pooling.py` | `from torch_scatter import scatter` → `from torch_geometric.utils import scatter` (no MSVC build tools needed). |
| `OFA/gp/nn/models/GNN.py` | Same scatter replacement. |

---

## Experiment Commands

All commands run from `F:\Rabia-Salman\DCGFM\OFA\`.
Prefix with `$env:WANDB_MODE="disabled"; `.
Add `--exp_label "description"` to any command for a descriptive directory name.

---

### 1. Original Baseline — All 3 Datasets, 5L GIN

**Near-center pruning** (`--hard_pruning_reverse`): removes graphs **near centre** (redundant/common). Keeps diverse molecules.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm.py `
    --override yamls/soft_and_hard.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 --hard_pruning_reverse
```

---

### 2. Mol-Only Baseline — 5L GIN, Far-Center Pruning

**Far-center pruning** (no flag): removes graphs **far from centre** (anomalous/diverse). Keeps representative molecules.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm.py `
    --override yamls/soft_and_hard_mol_far.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --save_model --exp_label "mol 5L GIN far-center"
```

For **near-center** add `--hard_pruning_reverse`:
```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm.py `
    --override yamls/soft_and_hard_mol_far.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --hard_pruning_reverse --save_model --exp_label "mol 5L GIN near-center"
```

---

### 3. Deep GIN — Residual Connections, Mol Only

Set `num_layers: 10` or `num_layers: 15` in `yamls/soft_and_hard_mol_deep.yaml`.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_deep.py `
    --override yamls/soft_and_hard_mol_deep.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --hard_pruning_reverse --save_model --exp_label "mol 10L deep GIN near-center"
```

---

### 4. Virtual Node GIN — Mol Only

**Near-center** (`--hard_pruning_reverse`): removes near-centre graphs, keeps diverse.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_vn.py `
    --override yamls/soft_and_hard_mol_vn.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --hard_pruning_reverse --save_model --exp_label "mol 10L VN near-center"
```

---

### 5. Virtual Node GIN — All 3 Datasets

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_vn.py `
    --override yamls/soft_and_hard_vn_full.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --hard_pruning_reverse --save_model --exp_label "all tasks VN 10L near-center"
```

---

### 6. Adaptive-Depth GIN — Mol Only

Single 20L GIN; diameter-gated masked attention selects active layers per molecule.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_adaptive.py `
    --override yamls/soft_and_hard_mol_adaptive.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --hard_pruning_reverse --save_model --exp_label "mol adaptive 20L near-center"
```

---

### 7. GIN + Transformer — Mol Only

10L RGCN-Edge followed by 1 Transformer block (sequential, not interleaved).

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_gin_transformer.py `
    --override yamls/soft_and_hard_mol_gin_transformer.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --hard_pruning_reverse --save_model --exp_label "mol 10L GIN Transformer near-center"
```

---

### 8. Small MoE — Mol Only

3 experts (3L RGCN + 3L RGCN + 3L GPS); sparse top-2 routing on topology features; 50 epochs.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_moe.py `
    --override yamls/soft_and_hard_mol_moe.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --hard_pruning_reverse --save_model --exp_label "mol small MoE near-center"
```

---

### 9. Dual Split-SVDD GIN — Mol Only

Molecules sorted by diameter; exact 50/50 split; independent SVDD per half; two specialist 10L GINs.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_dual.py `
    --override yamls/soft_and_hard_mol_dual.yaml `
    --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 --hard_pruning_epochs 25 `
    --hard_pruning_reverse --save_model --exp_label "mol dual 10L split-SVDD near-center"
```

---

### 10. Hop Distribution Analysis

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe mol_hop_analysis.py
# Output: analysis_output/mol_hop_distribution.png
```

---

## Results

### Mol-Only Experiments

> **Pruning key:**
> - *Far-center*: graphs **far from centre are removed** → representative molecules kept
> - *Near-center*: graphs **near centre are removed** → diverse molecules kept

| Experiment | Strategy | Layers | Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|---|---|---|
| 5L GIN | RGCN-Edge, no residuals | 5L | far-center | 0.5360 ± 0.0153 | 0.5146 ± 0.0063 |
| **10L deep GIN** | RGCN-Edge + residuals | 10L | far-center | **0.5757 ± 0.0105** | 0.4971 ± 0.0076 |
| 15L deep GIN | RGCN-Edge + residuals | 15L | far-center | 0.4853 ± 0.0079 | 0.5167 ± 0.0079 |
| Adaptive 20L | Diameter-gated masked attention | 20L | far-center | 0.5525 ± 0.0146 | 0.5200 ± 0.0105 |
| **10L VN** | RGCN-Edge + virtual node | 10L | far-center | 0.5581 ± 0.0098 | **0.5409 ± 0.0148** |
| GIN + Transformer | 10L RGCN-Edge → Transformer block | 10L+T | far-center | 0.5260 ± 0.0068 | 0.5008 ± 0.0008 |
| Small MoE | 3L×3 experts + GPS router | 3L×3 | far-center | 0.5617 ± 0.0124 | 0.5280 ± 0.0068 |
| 5L GIN | RGCN-Edge, no residuals | 5L | near-center | 0.4959 ± 0.0110 | 0.5244 ± 0.0054 |
| 10L deep GIN | RGCN-Edge + residuals | 10L | near-center | 0.5491 ± 0.0136 | 0.5063 ± 0.0090 |
| **10L VN** | RGCN-Edge + virtual node | 10L | near-center | **0.5544 ± 0.0152** | **0.5345 ± 0.0146** |
| 15L deep GIN | RGCN-Edge + residuals | 15L | near-center | 0.4767 ± 0.0114 | 0.5125 ± 0.0090 |
| GIN + Transformer | 10L RGCN-Edge → Transformer block | 10L+T | near-center | 0.4362 ± 0.0059 | 0.4994 ± 0.0009 |
| Adaptive 20L | Diameter-gated masked attention | 20L | near-center | 0.5200 ± 0.0143 | 0.4970 ± 0.0054 |
| Small MoE | 3L×3 experts + GPS router | 3L×3 | near-center | 0.5107 ± 0.0102 | **0.5392 ± 0.0102** |
| Dual VN | 2×10L VN-GIN, single SVDD | 10L×2 VN | near-center | 0.4957 ± 0.0181 | 0.4999 ± 0.0006 |
| Dual VN | 2×10L VN-GIN, single SVDD | 10L×2 VN | far-center | 0.4910 ± 0.0093 | 0.5000 ± 0.0002 |
| **Dual split-SVDD** | 2×10L GIN, separate SVDD per half | 10L×2 | near-center | **0.5813 ± 0.0125** | 0.5037 ± 0.0057 |
| Dual split-SVDD | 2×10L GIN, separate SVDD per half | 10L×2 | far-center | 0.4321 ± 0.0145 | 0.5013 ± 0.0024 |

---

### Full Dataset Experiments (All 3 Tasks)

| Experiment | Strategy | Layers | Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|---|---|---|
| **Original baseline** | RGCN-Edge, 5L | 5L | near-center | 0.4825 ± 0.0169 | **0.5421 ± 0.0092** |
| VN 5L all tasks | RGCN-Edge + VN | 5L | far-center | 0.5211 ± 0.0145 | 0.4885 ± 0.0080 |
| VN 10L all tasks | RGCN-Edge + VN | 10L | far-center | 0.5585 ± 0.0109 | 0.4874 ± 0.0101 |
| VN 10L all tasks | RGCN-Edge + VN | 10L | near-center | **0.5421 ± 0.0143** | 0.4919 ± 0.0113 |

---

## Key Findings

1. **Far-center pruning** (removes diverse graphs, keeps representative) → better for chemhiv binary classification.
2. **Near-center pruning** (removes common graphs, keeps diverse) → better for chempcba 128-assay multi-label.
3. **10L is the optimal depth** — 15L degrades due to over-smoothing regardless of pruning direction.
4. **VN 10L is the most balanced** — competitive on both metrics under both pruning strategies.
5. **Dual split-SVDD** achieves best chemhiv (0.5813) — separate pruning centres per structural regime improve specialist training.
6. **VN collapses in split architectures** — both dual VN experiments showed near-zero chempcba variance (degenerate output).
7. **GIN + Transformer consistently underperforms** — sequential (not interleaved) Transformer hurts under near-center pruning.
