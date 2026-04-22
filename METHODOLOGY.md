# DAP-GFM: Domain Adaptive Pruning for Graph Foundation Models
## Methodology

---

## 1. Problem Statement

DCGFM (KDD 2025) reduces GFM pre-training data to 30% while maintaining **aggregate**
accuracy. However, a per-domain breakdown of their Table 1 (50% pruning ratio) reveals
a systematic redistribution, not a uniform gain:

| Dataset | Whole Dataset | DCGFM 50% | Delta |
|---------|--------------|-----------|-------|
| ogbn-arxiv | 35.1 | 48.5 | **+13.4** |
| PubMed | 33.1 | 41.4 | **+8.3** |
| PCBA | 51.0 | 52.6 | **+1.6** |
| HIV | 53.2 | 48.8 | **-4.4** |
| FB15K237 | 55.5 | 53.6 | **-1.9** |
| WN18RR | 15.6 | 10.6 | **-5.0** |
| **AVG** | **40.6** | **42.6** | +2.0 |

The +2.0 AVG gain is entirely driven by citation improvements (+13.4 arxiv, +8.3 PubMed)
that mask degradation across three other domains. We introduce **Performance Variance
(PV) = std(per-domain accuracy)** as a primary evaluation metric alongside AVG —
this directly measures domain balance.

**Critical observation from the full table**: Across ALL methods tested, WN18RR is
systematically destroyed by any informativeness-based scoring. Soft Random (purely
random soft pruning) is the only method that does not hurt WN18RR (15.8 vs 15.6
whole). This reveals the root cause: KG link prediction structurally requires
repetitive relation-type triples that every scoring function treats as "uninformative"
and removes.

We identify three root causes:

1. **Global informativity threshold** — Deep SVDD scores all 180K subgraphs jointly.
   Structurally abundant citation graphs dominate the score distribution.

2. **Anomaly-only retention** — DCGFM keeps the top-α most anomalous subgraphs.
   This destroys the structural regularity that KG link prediction requires.

3. **Fixed global pruning ratio** — One β applied to all domains starves
   slow-learning domains (KG, molecule) that need proportionally more data.

---

## 1.1 On WN18RR's Inherently Low Accuracy

WN18RR whole dataset accuracy is 15.6 — only marginally above random baseline for
11-way relation classification (~9.1%). This reflects **fundamental task difficulty**
in the few-shot setting: WordNet's 11 semantic relations (hypernym, hyponym,
has_part, member_meronym, etc.) are fine-grained and hard to distinguish from few
examples with a generic text encoder (Sentence-BERT).

**We do not claim to fix WN18RR's absolute accuracy.** Our goal is narrower and
achievable: minimize the degradation from pruning by preserving relation-type balance.

**Root cause of pruning-induced collapse**: If any of the 11 WN18RR relation types
loses all its training examples in the pruned coreset, the model cannot learn that
relation type at all. We introduce **Relation-Type Retention (RTR)** — the fraction
of unique relation types represented in the pruned coreset — and add a hard
constraint to our KG pruning:

```
# Minimum coverage constraint for KG domain:
for each relation_type r in KG:
    ensure at least k_min = 10 examples of type r are in coreset
    → applied BEFORE any scoring-based selection fills remaining quota
```

This constraint is cheap (O(N) scan), domain-specific, and directly addresses the
WN18RR collapse. It is analogous to stratified sampling in class-imbalanced learning.

---

## 2. Proposed Methods

### Method 1: Domain-Oracle Hard Pruning (DOHP)
**Targets**: Root causes 1 and 2
**Complexity**: ~2-3 days
**Key insight from Table 1**: No single hard pruning algorithm is best across all
domains. The table reveals domain-specific optimal strategies:

| Domain | Best Hard Method in Table | Score | Reason |
|--------|--------------------------|-------|--------|
| Citation | DCGFM (SVDD) | 48.5/41.4 | Informativeness scoring works; citations are redundant |
| Molecule (HIV) | K-Center | **57.0** (beats whole 53.2) | Structural diversity helps molecular graphs |
| KG (WN18RR) | Soft Random | 15.8 (preserves whole) | Any scoring destroys relation-type balance |

**Algorithm**: Run domain-specific hard pruning:
```python
def domain_oracle_hard_prune(domain, subgraphs, ratio):
    if domain == "citation":
        # SVDD scoring with Goldilocks middle band
        scores = run_svdd(subgraphs)
        lo, hi = np.percentile(scores, [25, 75])
        return subgraphs[(scores >= lo) & (scores <= hi)]

    elif domain == "molecule":
        # Per-domain k-center for structural diversity
        return per_domain_kcenter(subgraphs, keep_ratio=1-ratio)

    elif domain == "KG":
        # Relation-type stratified random: ensure all types covered first,
        # then random sample remaining quota
        return stratified_random_kg(subgraphs, ratio, k_min=10)
```

**Why this is a contribution**: The domain-oracle is not cherry-picking — it is
a principled hypothesis that optimal data selection is domain-type-dependent.
We validate this hypothesis by ablating each domain strategy independently.

---

### Method 2: Per-Domain Adaptive InfoBatch (PDAIB)
**Targets**: Root cause 3
**Complexity**: ~3-4 days — modify `ofa_datasets_combine.py`

InfoBatch (ICLR 2024, oral) proved unbiased dynamic pruning via gradient rescaling
in **single-domain** settings. We extend it to multi-domain by maintaining separate
per-domain pruning ratios that adapt based on each domain's learning speed.

```python
# Per-domain EMA loss tracking
ema_loss[d] = 0.9 * ema_loss[d] + 0.1 * mean_loss_epoch[d]

# Adaptive ratio: slow-learning domain → prune less → give more data
speed[d] = (prev_ema[d] - ema_loss[d]) / (prev_ema[d] + 1e-8)
if speed[d] < θ_slow:
    prune_ratio[d] = max(β_min, prune_ratio[d] - δ)
elif speed[d] > θ_fast:
    prune_ratio[d] = min(β_max, prune_ratio[d] + δ)

# Gradient rescaling: unbiasedness preserved from InfoBatch
weight[i] = 1.0 / keep_prob[i]  # upweight kept samples
```

**Why it's novel**: InfoBatch multi-domain extension is not in the literature
(confirmed by search). This is the most technically rigorous contribution.

---

## 3. Experimental Design

### 3.1 Proxy Datasets

| Domain | Dataset | Size | Task | Metric |
|--------|---------|------|------|--------|
| Citation | Cora | 2,708 nodes | Node classification | Accuracy |
| Knowledge Graph | WN18RR | 40,943 entities | Link prediction | AUC |
| Molecule | HIV | 41,127 graphs | Graph classification | AUC |

```yaml
# OFA/yamls/dap_proxy.yaml
fs_sample_size: 3000
num_epochs: 10
batch_size: 128
hard_pruning_ratio: 0.5
prune_ratio: 0.5
```

### 3.2 Ablation Table

The ablation is structured in three tiers: baselines, domain-adaptive methods,
and causal methods. Stages 0 and 1 (causal environment construction + CSS scoring)
are training-free and add negligible compute. Stage 2 (TCS) adds ~25-30% overhead
to training — at proxy scale this is ~1 hr extra per run.

| # | Variant | Hard Pruning | Soft Pruning | Causal? | Est. Time |
|---|---------|-------------|-------------|---------|-----------|
| 1 | Whole Dataset | None | None | No | ~1 hr |
| 2 | DCGFM | Global SVDD top-α | Global InfoBatch | No | ~3 hrs |
| **Domain-Adaptive** | | | | | |
| 3 | Goldilocks | Global SVDD middle band | Global InfoBatch | No | ~3 hrs |
| 4 | DOHP only | Per-domain oracle strategy | Global InfoBatch | No | ~3 hrs |
| 5 | PDAIB only | Global SVDD top-α | Per-domain adaptive | No | ~3 hrs |
| 6 | DAP-Full | DOHP | PDAIB | No | ~3 hrs |
| **Causal** | | | | | |
| 7 | CSS only | CSS (Structural CIS + WLES) | Global InfoBatch | **Yes** | ~3.5 hrs |
| 8 | TCS-PDAIB only | Global SVDD top-α | PDAIB + gradient alignment | **Yes** | ~4 hrs |
| 9 | **Causal-DAP-Full** | CSS | TCS-PDAIB | **Yes** | ~4 hrs |
| 10 | **GDeR-DAP-Full** | GDeR (K prototypes) | TCS-PDAIB | **Yes** | ~5 hrs |

Total: ~31 hrs × 3 seeds = **~93 hrs**. Within 10 hrs/day budget across 4 weeks.

Variant #10 uses GDeR as a **learned replacement for CSS**: instead of WL-based
environment centroids, K prototype vectors are trained jointly with the GIN encoder.
Apply Goldilocks band on GDeR scores, then TCS-PDAIB for soft pruning.
This is the strongest possible system: learned causal scoring + adaptive domain pruning.

**What each comparison isolates**:
- #3 vs #2: does middle-band selection help over top-α alone?
- #4 vs #2: does domain-specific hard pruning strategy matter?
- #5 vs #2: does per-domain adaptive soft pruning help?
- #6 vs #4+#5: do hard+soft improvements compound?
- #7 vs #4: does causal CSS beat domain-oracle heuristic?
- #8 vs #5: does gradient conflict detection improve over EMA-only?
- #9 vs #6: does full causal system beat full domain-adaptive system?

### 3.3 Coreset Quality Metrics (Training-Free)

| Metric | Formula | What it shows |
|--------|---------|---------------|
| WL Coverage Rate (WCR) | `|WL(S)| / |WL(D)|` | Structural vocabulary retained |
| Domain Balance Score (DBS) | `−Σ p(d\|S) log p(d\|S)` | Domain entropy |
| **Relation-Type Retention (RTR)** | `|RelTypes(S_KG)| / |RelTypes(D_KG)|` | WN18RR degradation predictor |
| Performance Variance (PV) | `std(acc_d for d in domains)` | Primary evaluation metric |

RTR is the key diagnostic: show DCGFM's coreset loses WN18RR relation types;
DOHP's stratified random preserves all 11 types.

---

## 3.4 Full-Scale Hard Pruning Comparison (Training-Free, Across All Datasets)

**The key insight**: Hard pruning is purely a data selection step — it produces
a set of indices with no GFM training. We can run DCGFM's SVDD and our DOHP on
the full 180K OFA subgraphs AND on additional datasets from our project Table 1,
and compare the resulting coresets entirely without training.

**What runs**:
```
Full OFA subgraphs (180K):  arxiv + FB15K237 + mol   →  ~1-3 hrs each method
Extended datasets (OGB):    Cora, PubMed, WN18RR, HIV →  ~15-30 min each
```

Cora and PubMed already have `gen_data.py` in the codebase. OGB datasets (WN18RR,
HIV) download automatically. Each generates subgraphs using the same k-hop sampling
as OFA, then both hard pruning methods run on those subgraphs.

**Output**: A cross-dataset coreset quality table showing which domain types are
systematically harmed by global SVDD — covering our full project Table 1 without
a single epoch of GFM training.

---

## 3.5 How Results Compare Against Baseline — Complete Picture

There are three distinct comparison types in this project, each with a different
baseline reference. **It is critical not to mix these up.**

### Type 1: Coreset Quality Comparison (training-free)
**Baseline**: DCGFM's coreset (output of their hard pruning on same data)
**Our result**: DOHP coreset on same data
**How to compare**: Run both on identical input subgraphs, measure WCR/RTR/DBS
**Strength**: Full-scale, no training, mechanistically explains Table 1 failures
**Limitation**: Does not directly prove accuracy improvement

### Type 2: Proxy Training Comparison
**Baseline**: DCGFM re-run at proxy scale (our own run, same code, same proxy datasets)
**Our result**: All DAP-GFM variants at same proxy scale
**How to compare**: Direct numeric comparison, same conditions
**Strength**: Actual per-domain accuracy numbers under controlled conditions
**Limitation**: Proxy scale (3K samples, 10 epochs) — not full OFA benchmark

### Type 3: Context Reference (not a direct comparison)
**Baseline**: DCGFM paper's published Table 1 numbers (full scale, 8×A100)
**Our result**: NOT directly comparable — different scale, different compute
**How to use**: Cite as motivation ("DCGFM's own table shows WN18RR degradation")
**Critical note**: Never write "our method achieves X vs DCGFM's Y from their paper"
unless both are run at identical scale

---

## 3.6 Final Comparison Table Structure

The paper will have three result tables:

### Table A — Coreset Quality (Training-Free, Full Scale)
*Answers: does our pruning produce a structurally better dataset?*

| Method | WCR-Citation | WCR-KG | WCR-Mol | RTR (WN18RR) | DBS (↑) | IDD-KG |
|--------|-------------|--------|---------|--------------|---------|--------|
| No pruning (full D) | 100% | 100% | 100% | 11/11 | baseline | — |
| DCGFM hard prune | ? | ? | ? | **?/11** ← key | ? | ? |
| Goldilocks | ? | ? | ? | ?/11 | ? | ? |
| DOHP (ours) | ? | ? | ? | **11/11** (expected) | ? | ? |

RTR column is the headline finding: if DCGFM drops relation types and DOHP retains
all 11, this mechanistically proves why WN18RR collapses under DCGFM.

### Table B — Per-Domain Proxy Accuracy
*Answers: does our pruning produce a better-trained model?*

| Method | Cora (Acc↑) | WN18RR (AUC↑) | HIV (AUC↑) | AVG↑ | PV↓ |
|--------|------------|--------------|-----------|------|-----|
| Whole Dataset | — | — | — | — | — |
| DCGFM (proxy) | — | — | — | — | — |
| Goldilocks | — | — | — | — | — |
| DOHP only | — | — | — | — | — |
| PDAIB only | — | — | — | — | — |
| **DAP-Full** | — | — | — | — | — |

**PV (Performance Variance) is the primary metric** — lower = better domain balance.
AVG is secondary. A method that improves PV without hurting AVG is strictly better.

### Table C — Extended Coreset Analysis Across Dataset Types
*Answers: does the domain-collapse pattern generalize beyond OFA's three tasks?*

| Dataset | Domain Type | DCGFM RTR/WCR | DOHP RTR/WCR | Pattern |
|---------|------------|--------------|-------------|---------|
| ogbn-arxiv | Citation | ? | ? | SVDD helps? |
| PubMed | Citation | ? | ? | |
| Cora | Citation | ? | ? | |
| FB15K237 | KG | ? | ? | |
| WN18RR | KG | **?/11** | **11/11** | SVDD destroys KG |
| HIV | Molecule | ? | ? | k-center helps? |
| PCBA | Molecule | ? | ? | |

This table costs ~3-5 hrs total compute and shows the domain-collapse pattern is
systematic, not a WN18RR-specific accident.

### How Tables Connect to Arguments

| Argument | Evidence table | Strength without training |
|----------|---------------|--------------------------|
| DCGFM hides domain collapse in AVG | DCGFM Table 1 (cited) + Table B PV | Strong — their own numbers |
| Any scoring destroys KG relation types | Table A RTR + Table C | **Full-scale, no training** |
| Domain-specific strategy is necessary | Table A + Table C pattern | **Full-scale, no training** |
| DAP-Full improves per-domain balance | Table B | Proxy scale only |
| PDAIB reduces cross-seed variance | Table B std dev | Proxy scale only |

---

## 4. On Causality

> *Comment: Are we covering causality from the original proposal?*

**Honest answer: No, not in the current scope — and that is intentional.**

The original DAP-GFM proposal included Causal Invariance Score (CIS), Structural
Causal Models, and CIGA-style invariant subgraph extraction. These were removed
for two reasons:

1. **CIS is circular**: It requires model losses across environments, but hard
   pruning runs before training. A pre-trained proxy model would be needed to
   compute CIS, adding a full pre-training step before pruning begins.

2. **Feasibility**: True causal invariance testing (IRM, CIGA) requires environment
   partitioning and multi-environment invariance optimization — a research contribution
   in itself, not a 4-week add-on.

**What the current methods do (and don't) capture**:
- DOHP's WL-based structural coverage has superficial similarity to causal invariant
  features (both seek domain-independent structural patterns), but it is NOT
  causal in the formal sense — no interventions, no environment partitioning.
- PDAIB is purely loss-based; no causal component.

**If you want to add a lightweight causal element** (for framing, not full implementation):
The k-means structural environment partitioning from the original proposal
(Eq. 1: cluster subgraphs by WL vectors into K=5 environments) CAN be computed
without a model. You can then compute a structural CIS proxy:
```
structural_CIS(gᵢ) = 1 − std(WL_sim(gᵢ, centroid_k) for k in environments)
                         / mean(WL_sim(...))
```
This measures whether a subgraph's structural neighborhood is consistent across
environments — a rough, training-free proxy for causal invariance. Include it as
an **optional component** in DOHP's citation scoring.

**Recommendation**: Keep causality in the title and future work. Frame current work
as "structural domain-awareness as a necessary precondition for causal GFM pruning."
Do not claim causal validity for the current methods.

---

## 5. Future Extension: Replacing GIN with GDeR

> *For project extension toward publication*

DCGFM's hard pruning uses Deep SVDD with a **GIN encoder** — a single hypersphere
trained to enclose all "informative" subgraphs. The single-sphere assumption is a
known limitation: when data is multi-modal (multiple structurally distinct domains),
the sphere collapses to a poor compromise between domains.

**GDeR (NeurIPS 2024)**: "Safeguarding Efficiency, Balancing, and Robustness via
Prototypical Graph Pruning" directly addresses this. It replaces the single SVDD
center with **K learnable prototype vectors** on the hypersphere, each representing
a different structural mode of the data.

**How to integrate GDeR into DAP-GFM**:

1. Replace `GIN_Hard_Prune` in [hard_prune_module.py](OFA/hard_prune_module.py)
   with a GDeR-style model:
```python
class GDeR_Hard_Prune(pl.LightningModule):
    def __init__(self, nfeat, nhid, K_prototypes):
        # K learnable prototypes instead of single center
        self.prototypes = nn.Parameter(torch.randn(K_prototypes, nhid))
        self.encoder = GIN(nfeat, nhid)

    def forward(self, data):
        emb = self.encoder(data)
        # Score = distance to nearest prototype (not single center)
        dists = torch.cdist(emb, self.prototypes)
        return dists.min(dim=1).values   # anomaly score
```

2. Set K_prototypes per domain:
   - Citation: K=3 (few structural types: paper, citation-heavy, citation-sparse)
   - KG: K=11 (one per relation type in WN18RR)
   - Molecule: K=7 (approximate number of molecular scaffold types)

3. Apply Goldilocks band on resulting scores per domain.

**Expected gain**: Multi-prototype scoring avoids the mode-collapse problem of
single-sphere SVDD. For WN18RR specifically, setting K=11 (one prototype per
relation type) would naturally score each triple relative to its own relation-type
distribution — directly fixing the relation-type balance problem.

**Implementation estimate**: 1-2 additional weeks beyond current scope.
**Publication value**: GDeR + per-domain PDAIB is a complete, novel system
with NeurIPS/KDD-quality contributions if full-scale results are obtained.

---

## 6. Week-by-Week Timeline

### Week 0 (Prep)
- [ ] Download molecule dataset or confirm HIV (OGB) as proxy substitute
- [ ] Verify DCGFM 1-epoch proxy run completes without error
- [ ] Set up per-domain WandB metric logging

### Week 1 — Baselines + Coreset Analysis + Causal Environments
- [ ] Run variants #1 and #2 (whole dataset + DCGFM), 3 seeds (~6 hrs)
- [ ] Implement WCR, DBS, RTR, PV metrics (~2 hrs)
- [ ] **Stage 0**: WL histogram + k-means environment construction, verify ESE (~1 hr)
- [ ] Full-scale coreset comparison: DCGFM vs DOHP RTR/WCR (no training, ~3 hrs)

### Week 2 — Domain-Adaptive Hard Pruning + CSS Causal Hard Pruning
- [ ] Implement Goldilocks (2-line change) → run variant #3 (~3 hrs)
- [ ] Implement `stratified_random_kg()` + `per_domain_kcenter()` in `dap_prune.py`
- [ ] Run variant #4 (DOHP only, ~3 hrs)
- [ ] **Stage 1**: Implement structural CIS + WLES + CSS greedy selection in `dap_prune.py`
- [ ] Run variant #7 (CSS only, ~3.5 hrs)

### Week 3 — All Soft Pruning + Full Systems + GDeR
- [ ] Implement PDAIB (per-domain adaptive InfoBatch)
- [ ] Run variant #5 (PDAIB only, ~3 hrs)
- [ ] **Stage 2**: Implement TCS gradient alignment on top of PDAIB
- [ ] Run variant #8 (TCS-PDAIB, ~4 hrs)
- [ ] Run variant #6 (DAP-Full, ~3 hrs) and #9 (Causal-DAP-Full, ~4 hrs)
- [ ] **GDeR**: Add `GDeR_Hard_Prune` to `hard_prune_module.py` (K=5 prototypes, aligned to Stage 0 environments)
- [ ] Run variant #10 (GDeR-DAP-Full, ~5 hrs)

### Week 4 — Writing + Figures + Statistical Analysis
- [ ] Tables A, B, C fully populated
- [ ] Per-domain training curves, RTR bar charts, PV comparison plots
- [ ] Paper sections: Introduction, Related Work, Methodology, Results, Discussion
- [ ] Std dev across 3 seeds for all 9 variants

---

## 7. Files to Modify

| File | Change | Method |
|------|--------|--------|
| `OFA/run_cdm.py` | Modify `get_useful_indices()` — Goldilocks band | Method 1 |
| New: `OFA/dap_prune.py` | Domain-oracle strategies (SVDD+Goldilocks, k-center, stratified KG) | Method 1 |
| `OFA/run_cdm.py` | Replace `get_effective_indices()` to call domain-oracle | Method 1 |
| `OFA/ofa_datasets_combine.py` | Extend `InfoBatchMultiDataset` with per-domain EMA + adaptive ratio | Method 2 |
| `OFA/gp/lightning/module_template.py` | Add per-domain loss + PV logging | Evaluation |
| New: `OFA/yamls/dap_proxy.yaml` | Proxy experiment config | Experiments |
| New: `OFA/dap_causal.py` | WL histogram, k-means environments (Stage 0), CSS scoring (Stage 1) | Causal |
| `OFA/ofa_datasets_combine.py` | Add TCS gradient alignment monitor to PDAIB (Stage 2) | Causal |
| `OFA/hard_prune_module.py` | Add `GDeR_Hard_Prune` class alongside existing GIN | GDeR |

---

## 8. Future Work: GDeR Encoder + Full-Scale Validation

The current DAP-GFM system (all 9 variants including Causal-DAP-Full) is
validated at proxy scale. Two things remain as future work:

### 8.1 GDeR Encoder (now in current scope — see below)

DCGFM's hard pruning uses a GIN with a **single SVDD hypersphere** — a known
limitation when data is multi-modal. GDeR (NeurIPS 2024) replaces this with K
learnable prototype centers, one per structural mode.

In the current work, CSS already addresses this conceptually (via per-environment
WL scoring). GDeR provides a **learned** version of the same idea: instead of
WL-based environment centroids, the prototypes are optimized end-to-end.

**Integration**: Replace `GIN_Hard_Prune` in [hard_prune_module.py](OFA/hard_prune_module.py):
```python
class GDeR_Hard_Prune(pl.LightningModule):
    def __init__(self, nfeat, nhid, K_prototypes):
        self.prototypes = nn.Parameter(torch.randn(K_prototypes, nhid))
        self.encoder = GIN(nfeat, nhid)

    def forward(self, data):
        emb = self.encoder(data)
        dists = torch.cdist(emb, self.prototypes)
        return dists.min(dim=1).values   # score = distance to nearest prototype
```

Set K aligned to environments: K=5 (matching Stage 0 environments). Apply
Goldilocks band on scores per domain as in current CSS.

**Implementation**: ~1-2 weeks additional. Replaces WL-based CSS with a learned
equivalent — expected to be strictly stronger since prototypes are optimized.

### 8.2 Full-Scale OFA Training (~580 hrs on RTX 5090)

Running the full OFA pipeline (arxiv_fs + FB15K237_fs + mol_fs, 60K samples,
30 epochs) with Causal-DAP-Full is the path to top-venue publication.

**Publication path**:

| Stage | What's done | Venue target |
|-------|------------|-------------|
| Current (4 weeks) | All 9 variants at proxy scale + full-scale coreset analysis | NeurIPS/ICML Workshop |
| Extension 1 (+2 weeks) | GDeR encoder replaces CSS, proxy re-run | KDD 2026 / ICLR 2026 |
| Extension 2 (cloud ~$400) | Full-scale OFA training, all methods | KDD 2026 / NeurIPS 2026 |

The causal system (Stages 0, 1, 2) is complete within the current 4-week scope.
GDeR is a learned improvement over CSS, not a prerequisite.

---

## 9. Scope Statement

Full-scale validation requires ~450 hrs on RTX 5090 (estimated). We provide:
1. **Training-free**: Full-scale coreset quality (WCR, RTR, DBS) — runs in 1-3 hrs
2. **Proxy-scale**: Accuracy ablation on Cora + WN18RR + HIV, 10 epochs — ~3 hrs/run
3. **Context**: DCGFM paper's Table 1 as full-scale baseline motivation

All mechanistic claims (RTR explains WN18RR collapse; domain-specific strategies
are necessary) are validated at the coreset analysis level regardless of proxy results.