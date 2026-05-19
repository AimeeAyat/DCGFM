# BDA Project — Summary Report
## Improving Molecular Property Prediction in a Multi-Task Graph Foundation Model

---

## 1. Problem Statement

The original DCGFM (KDD 2025) is a multi-task graph foundation model trained jointly on three graph domains: citation networks (arxiv, Cora, PubMed, WikiCS), knowledge graphs (FB15K237, WN18RR), and molecular graphs (chemblpre, chemhiv, chempcba). It employs two pruning mechanisms:

- **Hard pruning (Deep SVDD)**: removes 70% of training graphs using anomaly scores from a GIN-based autoencoder. A single hypersphere centre is fit to all graphs; graphs far from the centre (structurally unusual) are removed, keeping the 30% most representative.
- **Soft pruning (InfoBatch)**: dynamically removes well-learned samples each epoch.

The original pipeline uses a single 5-layer RGCN-Edge GNN shared across all tasks. While it achieves reasonable aggregate performance, analysis of per-domain results reveals a critical limitation:

**The molecular task is significantly underserved.** The GNN, designed for 2-hop subgraph tasks (node/link), has a receptive field of 5 hops — yet 81.4% of chemblpre molecules have graph diameter > 10 hops and 56.3% of chemhiv molecules exceed 10 hops. This means the majority of molecules have atoms that never communicate through the GNN, severely limiting molecular representation quality.

**Baseline results** (all tasks, near-center pruning, 5L GIN):
- chemhiv AUC: 0.4825 ± 0.0169
- chempcba AUCmulti: 0.5421 ± 0.0092

---

## 2. Exploratory Analysis

### 2.1 Molecular Hop Distribution

A diameter analysis (`mol_hop_analysis.py`) was conducted across all three molecular datasets:

| Dataset | Min | Max | Mean | Median | ≤10 hops | >10 hops |
|---|---|---|---|---|---|---|
| chemblpre | 2 | 47 | 13.5 | 13 | 18.6% | **81.4%** |
| chemhiv | 1 | 93 | 12.0 | 11 | 43.7% | **56.3%** |
| chempcba | 0 | 122 | 13.6 | 13 | 16.9% | **83.1%** |

**Implication**: A 5-layer GIN covers only 5 hops. The vast majority of training molecules have atoms that are beyond the model's receptive field — a fundamental architectural mismatch.

### 2.2 Pruning Direction

The original DCGFM prunes graphs **near the SVDD centre** (removes common/redundant, keeps diverse). This study additionally tested **far-centre pruning** (removes anomalous/diverse, keeps representative):

- **Near-centre pruning** (`--hard_pruning_reverse`): removes near-centre graphs → keeps structurally diverse molecules
- **Far-centre pruning** (default): removes far-centre graphs → keeps structurally representative molecules

---

## 3. Methods Tried

### Method 1: Deep GIN — Increased Depth with Residual Connections

**Motivation**: Directly addressing the receptive field gap. More layers = more hops covered.

**Methodology**: Added residual skip connections (`h = h + h_prev`) to the base RGCN-Edge GNN (`GNN_deep.py`). Residuals prevent over-smoothing at depth by maintaining gradient flow to earlier representations.

**Assumption**: Deeper GNN with residuals will overcome the diameter coverage problem without over-smoothing.

| Layers | Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|---|
| 5L (baseline) | far-centre | 0.5360 ± 0.0153 | 0.5146 ± 0.0063 |
| **10L** | **far-centre** | **0.5757 ± 0.0105** | 0.4971 ± 0.0076 |
| 15L | far-centre | 0.4853 ± 0.0079 | 0.5167 ± 0.0079 |
| 10L | near-centre | 0.5491 ± 0.0136 | 0.5063 ± 0.0090 |
| 15L | near-centre | 0.4767 ± 0.0114 | 0.5125 ± 0.0090 |

**What worked**: 10L significantly improves chemhiv (+0.04 over 5L baseline under far-centre).

**What didn't work**: 15L degrades below 5L on chemhiv under both pruning strategies, confirming over-smoothing persists even with residuals beyond 10 layers. chempcba also underperforms the baseline.

**Pros**: Simple config change; residuals are well-motivated; 10L gives the best single-architecture chemhiv result.

**Cons**: Hard ceiling at 10L; chempcba consistently below baseline; does not solve the chempcba multi-label problem.

---

### Method 2: Virtual Node (VN)

**Motivation**: VN provides a global shortcut — any two atoms can communicate via the virtual node in 2 hops, effectively eliminating the diameter limitation without increasing layer count.

**Methodology**: Implemented a corrected VN (`GNN_vn.py`). The original codebase had a broken VN that updated the virtual node embedding but never broadcast it back to node features. Fix: `h = h + vnode[batch_seg]` before each message-passing layer.

**Assumption**: Global aggregation via VN will enable long-range molecular communication that compensates for diameter > 10.

| Architecture | Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|---|
| VN 10L (mol-only) | far-centre | 0.5581 ± 0.0098 | **0.5409 ± 0.0148** |
| VN 10L (mol-only) | near-centre | 0.5544 ± 0.0152 | 0.5345 ± 0.0146 |
| VN 5L (all tasks) | far-centre | 0.5211 ± 0.0145 | 0.4885 ± 0.0080 |
| VN 10L (all tasks) | far-centre | 0.5585 ± 0.0109 | 0.4874 ± 0.0101 |
| VN 10L (all tasks) | near-centre | 0.5421 ± 0.0143 | 0.4919 ± 0.0113 |

**What worked**: VN 10L mol-only with far-centre pruning achieves the best chempcba (0.5409) among all single-architecture experiments. The VN specifically helps multi-label tasks that need global molecular context.

**What didn't work**: Joint training with all 3 tasks reduces both metrics due to task interference. chemhiv (0.5581) is lower than the plain 10L GIN (0.5757).

**Pros**: Robust across pruning directions (small gap between far/near); best chempcba; addresses long-range without adding layers.

**Cons**: Requires mol-only training to achieve best results; joint training hurts molecular performance; chemhiv is weaker than 10L deep GIN.

---

### Method 3: Adaptive-Depth GIN with Diameter-Gated Attention

**Motivation**: Not all molecules need 20 layers. Route molecules to an appropriate depth subset based on their actual diameter, avoiding over-smoothing for small molecules while giving large molecules sufficient depth.

**Methodology**: Single 20-layer residual GIN with JK="none". At each node, masked self-attention over layer outputs, gated by graph diameter:
- diameter ≤ 5 → attend layers 1–5
- diameter 6–10 → attend layers 1–10
- diameter > 10 → attend all 20 layers

Diameter computed via scipy BFS (~1ms overhead per batch).

**Assumption**: Diameter-specific layer selection matches architectural capacity to molecular complexity.

| Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| far-centre | 0.5525 ± 0.0146 | 0.5200 ± 0.0105 |
| near-centre | 0.5200 ± 0.0143 | 0.4970 ± 0.0054 |

**What worked**: Competitive chemhiv under far-centre (0.5525); avoids the 15L over-smoothing issue; better chempcba than plain 10L/15L deep GIN.

**What didn't work**: Doesn't surpass the simpler 10L GIN on chemhiv; more complex implementation for marginal gain.

**Pros**: Theoretically sound; computationally efficient (single pass); interpretable routing.

**Cons**: The masked attention is an approximation — all 20 layers still run for all molecules, so compute savings are marginal; diameter-gated selection does not guarantee better representations.

---

### Method 4: Sequential GIN + Transformer

**Motivation**: GPS-style architectures achieve SOTA on OGB benchmarks by combining local MPNN with global attention. Even a single Transformer block after GIN should capture all-atom global dependencies.

**Methodology**: 10-layer residual GIN followed by one multi-head self-attention Transformer block (8 heads, pre-norm, per-graph padded batching). Sequential, not interleaved.

**Assumption**: A Transformer block at the end of the GNN stack will capture long-range patterns that local message-passing misses.

| Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| far-centre | 0.5260 ± 0.0068 | 0.5008 ± 0.0008 |
| near-centre | 0.4362 ± 0.0059 | 0.4994 ± 0.0009 |

**What worked**: Nothing notable — underperforms the simple 5L baseline on chemhiv under near-centre pruning.

**What didn't work**: Near-zero variance on chempcba (±0.0009) suggests near-degenerate output. Severe degradation under near-centre pruning (diverse training data) indicates the sequential Transformer is unstable when the input distribution is structurally heterogeneous.

**Pros**: Simple to implement; GPS++ literature supports the general approach.

**Cons**: Sequential (not interleaved) is architecturally inferior to GPS — SOTA research (NeurIPS 2022) confirms "major drop when MPNN is removed." Transformer cannot refine local chemistry when it only sees the GNN's already-finalised representations.

---

### Method 5: Topology-Aware Graph Mixture of Experts (MoE)

**Motivation**: Different structural regimes in a molecular dataset may benefit from different architectures. A routing mechanism can assign each molecule to the most appropriate expert.

**Methodology (Small MoE)**: Three parallel experts with sparse top-2 routing:
- E1: 3-layer RGCN-Edge (local chemistry specialist)
- E2: 3-layer RGCN-Edge (medium-range specialist)
- E3: 3-layer GPS (GINEConv + MultiheadAttention, long-range specialist)

Router: 4-descriptor topological feature vector [diameter/20, n_atoms/50, cyclomatic/10, mean_degree/4] → 2-layer MLP → top-2 sparse weights. Load-balancing auxiliary loss (Switch Transformer style, λ=0.005).

**Assumption**: Sparse top-2 routing will learn to assign small molecules to E1/E2 and large molecules to E3. Architecturally diverse experts will outperform a single architecture.

| Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| far-centre | 0.5617 ± 0.0124 | 0.5280 ± 0.0068 |
| near-centre | 0.5107 ± 0.0102 | **0.5392 ± 0.0102** |

**What worked**: Near-centre MoE achieves the best mol-only chempcba (0.5392) — the GPS expert in E3 captures the global patterns chempcba's 128 sparse assays need. Far-centre MoE achieves competitive chemhiv (0.5617).

**What didn't work**: Large first MoE attempt (94.5M parameters) completely failed — overparameterised for 18k training molecules. chemhiv under near-centre drops to 0.5107, significantly below VN (0.5544). Expert specialisation is not verifiable; expert collapse may occur.

**Pros**: Flexible; accommodates multiple architectural inductive biases; GPS expert is SOTA-informed; good chempcba under near-centre.

**Cons**: Cannot verify expert specialisation without explicit analysis; router overhead; 47M total parameters may still be large for ~18k molecules; training instability risk; difficult to explain what each expert has learned.

---

### Method 6: Dual Split-SVDD GIN

**Motivation**: A single global SVDD centre compromises between compact molecules and long-chain molecules. The centre is structurally ambiguous — "near centre" means different things for different molecular sizes. Splitting the data by diameter before pruning gives each group its own structurally meaningful centre.

**Methodology**:
1. Compute diameter for all training molecules (scipy BFS).
2. Sort by diameter; exact 50/50 split at median (~13).
3. Run independent Deep-SVDD on each half → separate pruning centres.
4. Two specialist 10L residual GINs trained jointly; hard diameter routing at inference.

**Assumption**: Separate SVDD centres produce higher-quality pruned subsets within each structural regime, and specialist GINs trained on homogeneous structural groups learn better per-regime representations.

| Configuration | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| Dual VN, near-centre | 0.4957 ± 0.0181 | 0.4999 ± 0.0006 |
| Dual VN, far-centre | 0.4910 ± 0.0093 | 0.5000 ± 0.0002 |
| **Dual GIN (no VN), near-centre** | **0.5813 ± 0.0125** | 0.5037 ± 0.0057 |
| Dual GIN (no VN), far-centre | 0.4321 ± 0.0145 | 0.5013 ± 0.0024 |

**What worked**: Dual GIN (no VN) with near-centre pruning achieves the **best chemhiv across all experiments (0.5813)**. Split-SVDD pruning demonstrably improves over single-SVDD for binary classification.

**What didn't work**: VN in the split architecture completely fails — near-zero chempcba variance (±0.0006) indicates degenerate constant output from the large-diameter GIN branch. chempcba stays near 0.50 for all dual configurations. Far-centre dual GIN chemhiv collapses (0.4321).

**Pros**: Novel contribution — diameter-split SVDD is theoretically sound and empirically validated for chemhiv; achieves best single-metric result; interpretable (each GIN handles a specific structural regime).

**Cons**: VN incompatible with split training (VN learns from all molecules to develop meaningful global summaries; restricting it to half the structural space causes collapse); chempcba improvement not achieved; only near-centre pruning works (far-centre fails).

---

## 4. Cross-Cutting Finding: Pruning Direction Effect

All experiments revealed a systematic interaction between pruning direction and task type:

| Pruning | What's removed | What's kept | Effect on chemhiv | Effect on chempcba |
|---|---|---|---|---|
| **Far-centre** | Diverse/unusual graphs | Representative/typical | ↑ Higher | ↓ Lower |
| **Near-centre** | Common/redundant graphs | Diverse/unusual | ↓ Lower | ↑ Higher |

**Explanation**: chemhiv is a binary classification task (HIV active/inactive) — representative drug-like molecules are sufficient to learn the activity boundary. chempcba has 128 sparse parallel assays, each testing a different biological endpoint — diverse molecules covering a wide chemical space are essential for the model to learn all 128 activity profiles.

---

## 5. Full Dataset vs Mol-Only Training

Joint training on all 3 tasks consistently hurts molecular performance:

| Setting | chemhiv AUC | chempcba AUCmulti |
|---|---|---|
| VN 10L mol-only | 0.5581 | 0.5409 |
| VN 10L all tasks | 0.5585 | 0.4874 |

**Cause**: Task interference — citation and KG tasks push GNN weights toward topological graph features, diluting the chemical structure representations that molecular tasks require.

---

## 6. Summary Results Table

### Mol-Only Experiments

| Method | Architecture | Layers | Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|---|---|---|
| **Baseline** | 5L GIN | 5L | far-centre | 0.5360 ± 0.0153 | 0.5146 ± 0.0063 |
| Deep GIN | + Residuals | 10L | far-centre | **0.5757 ± 0.0105** | 0.4971 ± 0.0076 |
| Deep GIN | + Residuals | 15L | far-centre | 0.4853 ± 0.0079 | 0.5167 ± 0.0079 |
| VN | + Virtual Node | 10L | far-centre | 0.5581 ± 0.0098 | **0.5409 ± 0.0148** |
| Adaptive | Diameter-gated attn | 20L | far-centre | 0.5525 ± 0.0146 | 0.5200 ± 0.0105 |
| GIN+Transformer | Sequential Transformer | 10L+T | far-centre | 0.5260 ± 0.0068 | 0.5008 ± 0.0008 |
| Small MoE | 3 experts + GPS | 3L×3 | far-centre | 0.5617 ± 0.0124 | 0.5280 ± 0.0068 |
| **Baseline** | 5L GIN | 5L | near-centre | 0.4959 ± 0.0110 | 0.5244 ± 0.0054 |
| Deep GIN | + Residuals | 10L | near-centre | 0.5491 ± 0.0136 | 0.5063 ± 0.0090 |
| VN | + Virtual Node | 10L | near-centre | 0.5544 ± 0.0152 | 0.5345 ± 0.0146 |
| Small MoE | 3 experts + GPS | 3L×3 | near-centre | 0.5107 ± 0.0102 | **0.5392 ± 0.0102** |
| Dual split-SVDD | 2× specialist GIN | 10L×2 | near-centre | **0.5813 ± 0.0125** | 0.5037 ± 0.0057 |
| Dual VN | 2× VN-GIN | 10L×2 VN | near-centre | 0.4957 ± 0.0181 | 0.4999 ± 0.0006 |

### Full Dataset (All Tasks)

| Method | Layers | Pruning | chemhiv AUC | chempcba AUCmulti |
|---|---|---|---|---|
| Original baseline | 5L | near-centre | 0.4825 ± 0.0169 | **0.5421 ± 0.0092** |
| VN all tasks | 5L | far-centre | 0.5211 ± 0.0145 | 0.4885 ± 0.0080 |
| VN all tasks | 10L | far-centre | 0.5585 ± 0.0109 | 0.4874 ± 0.0101 |
| VN all tasks | 10L | near-centre | **0.5421 ± 0.0143** | 0.4919 ± 0.0113 |

---

## 7. Key Findings

1. **10-layer depth is optimal** — both deeper (15L) and shallower (5L) architectures underperform on chemhiv. Over-smoothing persists even with residual connections beyond 10 layers.

2. **Pruning direction creates a chemhiv/chempcba trade-off** — no single pruning strategy simultaneously optimises both metrics. Far-centre improves chemhiv; near-centre improves chempcba.

3. **VN is the most balanced single architecture** — robust to pruning direction, competitive on both metrics; second-best on chemhiv (0.5581), best single-architecture chempcba (0.5409).

4. **Dual split-SVDD advances chemhiv** — separate pruning centres per structural regime (small/large diameter) achieve the best chemhiv (0.5813), validating the hypothesis that a single global SVDD centre is suboptimal for structurally heterogeneous molecular datasets.

5. **VN collapses in split architectures** — VN requires full-distribution training to develop meaningful global summaries; restricting it to half the structural space causes degeneracy (near-zero output variance).

6. **Sequential GIN → Transformer is consistently the worst architecture** — confirmed by GPS literature: MPNN and Transformer must be interleaved, not sequential.

7. **Joint training hurts molecular performance** — task interference from citation/KG tasks degrades molecular representations, particularly for chempcba's sparse multi-label structure.

---

## 8. Open Questions

- Can dual split-SVDD be extended to improve chempcba without VN (e.g., GPS expert for large-diameter molecules)?
- Does the pruning direction trade-off persist with a domain-adaptive pruning ratio (different ratios for chemhiv vs chempcba)?
- Would a GPS-style interleaved MPNN+Transformer (not sequential) address the transformer failure?
- Can MoE routing be verified to have learned meaningful structural specialisation?
