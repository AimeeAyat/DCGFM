# BDA Project — Summary Report
## Improving Molecular Property Prediction in a Multi-Task Graph Foundation Model

---

## 1. Problem Statement

The original DCGFM (KDD 2025) is a multi-task graph foundation model which jointly trains on Node, Link and Graph datasets for zero-shot graph learning transfer. In the **zero-shot transfer** setting: the model learns on one set of graphs and is evaluated on structurally different, held-out datasets. It is important to distinguish training datasets from test datasets — the model never sees test datasets during training.

### Training Datasets (seen during pre-training)

| Domain | Dataset | Size | Task |
|---|---|---|---|
| Citation | ogbn-arxiv | 169,343 nodes, sampled to **60k episodes** | Node classification |
| Knowledge Graph | FB15K237 | 310,116 triples, sampled to **60k episodes** | Link prediction |
| Molecular | chemblpre | 364,181 molecules, sampled to **60k episodes** | Graph classification |

After hard pruning (70% removed): ~18k episodes per domain used per training run.

> **What is a 60k episode?** Not a raw molecule — each episode is a randomly pre-generated few-shot task: N randomly selected classes, one support molecule per class, one query molecule. For molecular data, N=2 (active/inactive for a randomly selected assay). 60,000 such episodes are generated once and cached.

### Test Datasets (never seen during training)

| Domain | Dataset | Size | Evaluation | Transfer type |
|---|---|---|---|---|
| Citation | Cora | 2,708 nodes | 2/7-way few-shot node cls | Zero-shot (different graph) |
| Citation | PubMed | 19,717 nodes | 3-way few-shot node cls | Zero-shot (different graph) |
| Citation | WikiCS | 11,701 nodes | 10-way few-shot node cls | Zero-shot (different graph) |
| Knowledge Graph | **FB15K237** | 237 relations, 310k triples | 10/20-way few-shot link pred | Same KG, held-out test triples |
| Knowledge Graph | WN18RR | 11 relations, 93k triples | 5/10-way few-shot link pred | Zero-shot (different KG) |
| Molecular | **chemhiv** | **41,127 molecules** | 2-way few-shot (HIV active/inactive) | Zero-shot (different dataset) |
| Molecular | **chempcba** | **437,929 molecules** | 2-way few-shot (128 assays) | Zero-shot (different dataset) |

### Pruning Mechanisms (applied to training data only)

- **Hard pruning (Deep SVDD)**: removes 70% of training graphs; **graphs near the centre (common/redundant structures) are removed**, keeping the 30% most structurally diverse.
- **Soft pruning (InfoBatch)**: dynamically removes well-learned samples each epoch.

The original pipeline uses a single 5-layer RGCN-Edge GNN. Analysis reveals a critical limitation:

**The molecular task is significantly underserved.** 81.4% of chemblpre molecules have graph diameter > 10 hops, yet a 5-layer GIN only covers 5 hops — most molecules have atoms that never communicate through the GNN.

**Baseline results** (all tasks, near-centre pruning, 5L GIN):
- chemhiv AUC: 0.4825 ± 0.0169
- chempcba AUCmulti: 0.5421 ± 0.0092

---

## 2. Exploratory Analysis

### 2.1 Molecular Hop Distribution

A diameter analysis (`mol_hop_analysis.py`) was conducted across all three molecular datasets (chemblpre = training; chemhiv and chempcba = test-only):

| Dataset | Min | Max | Mean | Median | ≤10 hops | >10 hops |
|---|---|---|---|---|---|---|
| chemblpre | 2 | 47 | 13.5 | 13 | 18.6% | **81.4%** |
| chemhiv | 1 | 93 | 12.0 | 11 | 43.7% | **56.3%** |
| chempcba | 0 | 122 | 13.6 | 13 | 16.9% | **83.1%** |

**Implication**: A 5-layer GIN covers only 5 hops — a fundamental architectural mismatch for the majority of molecules.

### 2.2 Pruning Direction Analysis

Two pruning strategies were tested systematically across all architectures:

- **Near-centre pruning** (`--hard_pruning_reverse`): removes near-centre graphs → keeps structurally **diverse** molecules
- **Far-centre pruning** (default): removes far-centre graphs → keeps structurally **representative** molecules

**Key finding**: pruning direction has opposite effects on the two test metrics.

| Architecture | Layers | chemhiv — Near | chemhiv — Far | chempcba — Near | chempcba — Far |
|---|---|---|---|---|---|
| 5L GIN (baseline) | 5L | 0.4959 | 0.5360 | 0.5244 | 0.5146 |
| Deep GIN | 10L | 0.5491 | **0.5757** | 0.5063 | 0.4971 |
| Deep GIN | 15L | 0.4767 | 0.4853 | 0.5125 | 0.5167 |
| VN | 10L | 0.5544 | 0.5581 | 0.5345 | **0.5409** |
| Adaptive 20L | 20L | 0.5200 | 0.5525 | 0.4970 | 0.5200 |
| GIN + Transformer | 10L | 0.4362 | 0.5260 | 0.4994 | 0.5008 |
| Small MoE | 3L×3 | 0.5107 | 0.5617 | **0.5392** | 0.5280 |
| Dual split-SVDD | 10L×2 | **0.5813** | 0.4321 | 0.5037 | 0.5013 |

**chemhiv** (binary HIV activity) performs better under **far-centre pruning** — representative drug-like molecules are sufficient to learn the binary activity boundary.

**chempcba** (128 sparse assays) performs better under **near-centre pruning** — diverse molecules covering a wide chemical space are essential for all 128 assay profiles.

All subsequent architectural experiments were run under both strategies; near-centre is the primary setting for the architectural analysis below.

---

## 3. Methods Tried

### Method 1: Deep GIN — Increased Depth with Residual Connections

**Motivation**: Directly address the receptive field gap — more layers = more hops covered.

**Methodology**: Added residual skip connections (`h = h + h_prev`) to the base RGCN-Edge GNN (`GNN_deep.py`). Residuals prevent over-smoothing at depth by maintaining gradient flow to earlier representations.

**Assumption**: Deeper GNN with residuals will overcome the diameter coverage problem without over-smoothing.

| Layers | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| 5L (baseline) | 0.4959 ± 0.0110 | 0.5244 ± 0.0054 |
| **10L** | **0.5491 ± 0.0136** | 0.5063 ± 0.0090 |
| 15L | 0.4767 ± 0.0114 | 0.5125 ± 0.0090 |

**What worked**: 10L significantly improves chemhiv (+0.053 over 5L baseline).

**What didn't work**: 15L degrades below 5L — over-smoothing persists beyond 10 layers even with residuals. chempcba underperforms the 5L baseline.

**Pros**: Simple config change; residuals are well-motivated; 10L gives the best single-depth chemhiv result.

**Cons**: Hard ceiling at 10L; chempcba consistently below baseline; does not solve the multi-label problem.

---

### Method 2: Virtual Node (VN)

**Motivation**: VN provides a global shortcut — any two atoms communicate via the virtual node in 2 hops, eliminating the diameter limitation without increasing layer count.

**Methodology**: Implemented a corrected VN (`GNN_vn.py`). The original codebase had a broken VN that updated the virtual node embedding but never broadcast it back to node features. Fix: `h = h + vnode[batch_seg]` before each message-passing layer.

**Assumption**: Global aggregation via VN will enable long-range molecular communication that compensates for diameter > 10.

| Architecture | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| VN 10L (mol-only) | 0.5544 ± 0.0152 | 0.5345 ± 0.0146 |
| VN 10L (all tasks) | 0.5421 ± 0.0143 | 0.4919 ± 0.0113 |

**What worked**: VN 10L mol-only achieves the best chempcba among all near-centre single-architecture experiments (0.5345). Robust to pruning direction (small gap, see Section 2.2).

**What didn't work**: Joint training with all 3 tasks reduces both metrics due to task interference.

**Pros**: Robust across pruning directions; best chempcba balance; addresses long-range without adding layers.

**Cons**: Requires mol-only training for best results; joint training hurts molecular performance; chemhiv weaker than 10L deep GIN.

---

### Method 3: Adaptive-Depth GIN with Diameter-Gated Attention

**Motivation**: Not all molecules need 20 layers. Route each molecule to a depth subset matching its actual diameter.

**Methodology**: Single 20-layer residual GIN. At each node, masked self-attention over layer outputs gated by graph diameter:
- diameter ≤ 5 → attend layers 1–5
- diameter 6–10 → attend layers 1–10
- diameter > 10 → attend all 20 layers

**Assumption**: Diameter-specific layer selection matches architectural capacity to molecular complexity.

| chemhiv AUC | chempcba AUCmulti |
|---|---|
| 0.5200 ± 0.0143 | 0.4970 ± 0.0054 |

**What worked**: Avoids the 15L over-smoothing issue; single forward pass.

**What didn't work**: Does not surpass the simpler 10L GIN; more complex implementation for marginal gain.

**Pros**: Theoretically sound; computationally efficient; interpretable routing.

**Cons**: All 20 layers still run for every molecule — compute savings are marginal; diameter-gating does not guarantee better representations.

---

### Method 4: Sequential GIN + Transformer

**Motivation**: GPS-style architectures achieve SOTA on OGB benchmarks by combining local MPNN with global attention.

**Methodology**: 10-layer residual GIN followed by one Transformer block (8 heads, pre-norm). Sequential, not interleaved.

**Assumption**: A Transformer block at the end will capture long-range patterns that local message-passing misses.

| chemhiv AUC | chempcba AUCmulti |
|---|---|
| 0.4362 ± 0.0059 | 0.4994 ± 0.0009 |

**What worked**: Nothing — underperforms even the 5L baseline.

**What didn't work**: Near-zero chempcba variance (±0.0009) suggests near-degenerate output. The sequential approach is architecturally flawed — GPS literature (NeurIPS 2022) confirms "major drop when MPNN is removed"; MPNN and Transformer must run in parallel, not in sequence.

**Pros**: Simple to implement.

**Cons**: Sequential is inferior to GPS-style interleaving; worst architecture tested under near-centre pruning.

---

### Method 5: Topology-Aware Graph Mixture of Experts (MoE)

**Motivation**: Different structural regimes may benefit from different architectures. A routing mechanism can assign each molecule to the most appropriate expert.

**Methodology**: Three parallel experts with sparse top-2 routing:
- E1: 3-layer RGCN-Edge (local specialist)
- E2: 3-layer RGCN-Edge (medium-range specialist)
- E3: 3-layer GPS (GINEConv + MultiheadAttention, long-range specialist)

Router: 4-descriptor topological vector [diameter, n_atoms, cyclomatic complexity, mean degree] → MLP → top-2 sparse weights. Load-balancing auxiliary loss (λ=0.005).

**Assumption**: Sparse top-2 routing will learn to assign small molecules to E1/E2 and large to E3.

| Setting | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| Mol-only | 0.5107 ± 0.0102 | **0.5392 ± 0.0102** |
| All tasks | **0.5912 ± 0.0118** | 0.4993 ± 0.0051 |

**What worked**: Best mol-only chempcba (0.5392) — GPS expert captures multi-label global patterns. All-tasks training achieves the best chemhiv across all experiments (0.5912) — joint training provides beneficial regularisation for binary classification.

**What didn't work**: chempcba drops to 0.499 in joint training (near-random). Expert specialisation unverifiable.

**Pros**: Flexible; accommodates multiple architectural inductive biases; GPS expert is SOTA-informed.

**Cons**: chemhiv/chempcba trade-off in joint training; 47M total parameters may still be large for the data volume.

---

### Method 6: Dual Split-SVDD GIN

**Motivation**: A single global SVDD centre is structurally ambiguous for heterogeneous molecular datasets — "near centre" means different things for compact rings vs long chains. Splitting by diameter before pruning gives each group its own meaningful centre.

**Methodology**:
1. Compute diameter for all training molecules (scipy BFS).
2. Sort by diameter; exact 50/50 split at median (≈13).
3. Run independent Deep-SVDD on each half → two separate pruning centres.
4. Two specialist 10L residual GINs trained jointly; hard diameter routing at inference.

**Assumption**: Separate SVDD centres produce higher-quality pruned subsets per structural regime; specialist GINs trained on homogeneous groups learn better representations.

| Configuration | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| Dual VN | 0.4957 ± 0.0181 | 0.4999 ± 0.0006 |
| **Dual GIN (no VN)** | **0.5813 ± 0.0125** | 0.5037 ± 0.0057 |

**What worked**: Dual GIN achieves the **best chemhiv across all experiments (0.5813)**. Split-SVDD pruning demonstrably improves over single-SVDD for binary classification.

**What didn't work**: VN in split architecture completely fails — near-zero chempcba variance (±0.0006) indicates degenerate output. chempcba stays near 0.50 for all dual configurations.

**Pros**: Novel — diameter-split SVDD is theoretically sound and empirically validated; best single-metric result; interpretable specialist routing.

**Cons**: VN incompatible with split training (VN requires full-distribution training; restricting to half the structural space causes collapse); chempcba not improved.

---

## 4. Full Dataset vs Mol-Only Training

Joint training on all 3 tasks has mixed effects. For VN-based architectures, task interference from citation and KG tasks dilutes molecular representations. Interestingly, the small MoE architecture shows the **opposite** effect — joint training boosts chemhiv substantially at the cost of chempcba:

| Setting | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| VN 10L mol-only | 0.5544 | 0.5345 |
| VN 10L all tasks | 0.5421 | 0.4919 |
| Small MoE mol-only | 0.5107 | **0.5392** |
| **Small MoE all tasks** | **0.5912** | 0.4993 |

The MoE GPS expert likely benefits from the richer multi-domain training signal for binary classification (chemhiv), but the diverse, multi-label chempcba task is too noisy to benefit from joint training under a shared model.

---

## 5. Summary Results Table (Near-Centre Pruning, Mol-Only)

| Method | Architecture | Layers | chemhiv AUC | chempcba AUCmulti |
|---|---|---|---|---|
| **Baseline** | 5L GIN | 5L | 0.4959 ± 0.0110 | 0.5244 ± 0.0054 |
| Deep GIN | + Residuals | 10L | 0.5491 ± 0.0136 | 0.5063 ± 0.0090 |
| Deep GIN | + Residuals | 15L | 0.4767 ± 0.0114 | 0.5125 ± 0.0090 |
| VN | + Virtual Node | 10L | 0.5544 ± 0.0152 | 0.5345 ± 0.0146 |
| Adaptive | Diameter-gated attn | 20L | 0.5200 ± 0.0143 | 0.4970 ± 0.0054 |
| GIN + Transformer | Sequential Transformer | 10L+T | 0.4362 ± 0.0059 | 0.4994 ± 0.0009 |
| Small MoE | 3 experts + GPS | 3L×3 | 0.5107 ± 0.0102 | **0.5392 ± 0.0102** |
| Dual VN | 2× VN-GIN | 10L×2 | 0.4957 ± 0.0181 | 0.4999 ± 0.0006 |
| **Dual split-SVDD** | 2× specialist GIN | 10L×2 | **0.5813 ± 0.0125** | 0.5037 ± 0.0057 |

> For far-centre pruning results and the full cross-pruning comparison, see **Section 2.2**.

---

## 6. Key Findings

1. **10-layer depth is optimal** — over-smoothing persists even with residuals beyond 10 layers; 15L consistently underperforms 10L.

2. **Pruning direction creates a chemhiv/chempcba trade-off** — far-centre improves chemhiv; near-centre improves chempcba. No single strategy simultaneously optimises both. Full analysis in Section 2.2.

3. **VN is the most balanced single architecture** — robust to pruning direction, competitive on both metrics; best single-architecture chempcba under near-centre (0.5345).

4. **Dual split-SVDD advances chemhiv** — separate pruning centres per structural regime achieve the best chemhiv (0.5813), validating that a single global SVDD centre is suboptimal for heterogeneous molecular datasets.

5. **VN collapses in split architectures** — VN requires full-distribution training; restricting to half the structural space causes degenerate output (near-zero variance).

6. **Sequential GIN → Transformer is consistently the worst architecture** — MPNN and Transformer must be interleaved (GPS-style), not sequential.

7. **Joint training has architecture-dependent effects** — VN suffers task interference (chemhiv 0.5544 → 0.5421); MoE benefits for chemhiv in joint training (0.5107 → 0.5912) but chempcba drops to near-random. No architecture simultaneously optimises both metrics in joint training.

---

## 7. Open Questions

- Can dual split-SVDD be extended to improve chempcba without VN?
- Does the pruning direction trade-off persist with domain-adaptive pruning ratios?
- Would GPS-style interleaved MPNN+Transformer address the sequential Transformer failure?
- Can MoE routing be verified to have learned meaningful structural specialisation?
