# BDA Experiments — DCGFM Extension

This document covers all experiments, run commands, and new files added on top of
the original [DCGFM](https://github.com/Yuhan1i/DCGFM) codebase for the BDA
project.  The original paper code is unchanged; every new feature lives in a
separate file and is injected at runtime via namespace patching.

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
| `OFA/gp/nn/models/GNN_deep.py` | `DeepMultiLayerMessagePassing` — base class that adds residual skip connections (`h = h + prev_h`) from layer 1 onward. Prevents over-smoothing when stacking 10+ layers. |
| `OFA/gp/nn/models/GNN_vn.py` | `VNMultiLayerMessagePassing` — corrected virtual node implementation. The original `GNN.py` VN updated the virtual node each layer but **never broadcast it back** to node features, making it inert. This file adds the missing broadcast step: `h = h + vnode[batch_seg]` before each layer. |

### Model Classes

| File | Purpose |
|---|---|
| `OFA/models/model_deep.py` | `PyGRGCNEdgeDeep` — RGCN-Edge conv inheriting from `DeepMultiLayerMessagePassing`. Drop-in replacement for `PyGRGCNEdge`. |
| `OFA/models/model_vn.py` | `PyGRGCNEdgeVN` — RGCN-Edge conv with correct virtual node broadcast. Drop-in replacement for `PyGRGCNEdge`. |
| `OFA/models/model_adaptive.py` | `AdaptiveGINModel` — single 20-layer residual GIN where each molecule attends only over the layer subset matching its graph diameter: ≤5 hops → layers 1–5, 6–10 hops → layers 1–10, >10 hops → all 20. Uses `MaskedSingleHeadAtt` to gate attention per molecule. Replaces `BinGraphAttModel` entirely. |

### Runner Scripts

Each runner patches one name in `run_cdm`'s namespace and calls the original
`main()` — the original `run_cdm.py` is never modified.

| File | Purpose |
|---|---|
| `OFA/run_cdm_deep.py` | Injects `PyGRGCNEdgeDeep` → runs the residual deep GIN experiment. |
| `OFA/run_cdm_vn.py` | Injects `PyGRGCNEdgeVN` → runs the virtual node experiment. Also patches `dict_res_summary` to write `results.json` to the experiment directory at the end of training. |
| `OFA/run_cdm_adaptive.py` | Injects `AdaptiveGINModel` → runs the adaptive-depth experiment. |

### Analysis & Configs

| File | Purpose |
|---|---|
| `OFA/mol_hop_analysis.py` | Computes graph diameter (longest shortest path) for chemblpre, chemhiv, chempcba via scipy BFS. Outputs histogram showing how many molecules fall in the 2/4/6/10-hop buckets. Results cached to `analysis_output/mol_diameters.pkl`. |
| `OFA/yamls/soft_and_hard_mol_far.yaml` | mol-only, 5 layers, far-center pruning. |
| `OFA/yamls/soft_and_hard_mol_deep.yaml` | mol-only, 15 layers, residual GIN. |
| `OFA/yamls/soft_and_hard_mol_vn.yaml` | mol-only, 10 layers, virtual node. |
| `OFA/yamls/soft_and_hard_mol_adaptive.yaml` | mol-only, 20 layers, adaptive depth. |
| `OFA/yamls/soft_and_hard_vn_full.yaml` | All 3 datasets, virtual node (configurable layers). |
| `OFA/saved_exp/README.md` | Full results table across all 12 experiments. |

### Bug Fixes in Original Files

| File | Fix |
|---|---|
| `OFA/run_cdm.py` | Added `_fwd` suffix to hard-pruning cache key when `hard_pruning_reverse=False`, preventing cache collisions between near-center and far-center experiments. Also changed cache validity check from "does `_0.pkl` exist?" to "do **all** task pkl files exist?" — prevents partial cache from a mol-only run being loaded for a 3-task run. |
| `OFA/gp/nn/pooling.py` | Replaced `from torch_scatter import scatter` → `from torch_geometric.utils import scatter` (no C++ build tools required). |
| `OFA/gp/nn/models/GNN.py` | Same scatter replacement. |

---

## Experiment Commands

All commands are run from `F:\Rabia-Salman\DCGFM\OFA\`.
Prefix every command with `$env:WANDB_MODE="disabled"; `.

---

### 1. Original Baseline — Hard 70% + Soft 30%, All Datasets

Prunes **near center** (removes redundant/common graphs).

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm.py `
    --override yamls/soft_and_hard.yaml `
    --hard_pruning_mode hard_prune_api `
    --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 `
    --hard_pruning_reverse `
    --hard_pruning_epochs 25
```

---

### 2. Far-Center Pruning — Default GIN, mol only

Prunes **far from center** (removes outlier/anomalous graphs).
No `--hard_pruning_reverse` flag = far-center mode.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm.py `
    --override yamls/soft_and_hard_mol_far.yaml `
    --hard_pruning_mode hard_prune_api `
    --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 `
    --hard_pruning_epochs 25
```

---

### 3. Deep GIN — Residual Connections, mol only

10 or 15 layers with residual skip connections.
Edit `num_layers` in `yamls/soft_and_hard_mol_deep.yaml` to switch depth.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_deep.py `
    --override yamls/soft_and_hard_mol_deep.yaml `
    --hard_pruning_mode hard_prune_api `
    --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 `
    --hard_pruning_epochs 25
```

---

### 4. Adaptive-Depth GIN — Diameter-gated attention, mol only

Single 20-layer GIN; molecules route to layers 1–5 / 1–10 / 1–20 based on graph diameter.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_adaptive.py `
    --override yamls/soft_and_hard_mol_adaptive.yaml `
    --hard_pruning_mode hard_prune_api `
    --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 `
    --hard_pruning_epochs 25
```

---

### 5. Virtual Node GIN — mol only

10-layer RGCN-Edge with correctly-wired virtual node.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_vn.py `
    --override yamls/soft_and_hard_mol_vn.yaml `
    --hard_pruning_mode hard_prune_api `
    --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 `
    --hard_pruning_epochs 25
```

---

### 6. Virtual Node GIN — All Datasets (5 layers)

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm_vn.py `
    --override yamls/soft_and_hard_vn_full.yaml `
    --hard_pruning_mode hard_prune_api `
    --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 `
    --hard_pruning_epochs 25
```

To run with 10 layers, set `num_layers: 10` in `yamls/soft_and_hard_vn_full.yaml`.

---

### 7. Hop Distribution Analysis (run once)

Computes and plots molecular graph diameter distribution.
Results cached — subsequent runs are instant.

```powershell
F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe mol_hop_analysis.py
# Output: analysis_output/mol_hop_distribution.png
```

---

## Results Summary

See [`OFA/saved_exp/README.md`](OFA/saved_exp/README.md) for the full table.

| Experiment | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| Baseline 5L near-center (all datasets) | 0.4825 | **0.5421** |
| Far-center pruning 5L (mol only) | 0.5360 | 0.5146 |
| Deep GIN 10L + residuals (mol only) | **0.5757** | 0.4971 |
| Deep GIN 15L + residuals (mol only) | 0.4853 | 0.5167 |
| Adaptive-depth 20L (mol only) | 0.5525 | 0.5200 |
| VN 10L (mol only) | 0.5581 | 0.5409 |
| VN 5L all datasets | 0.5211 | — |
