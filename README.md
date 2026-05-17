# Advancing Graph Foundation Models: A Data-Centric Perspective

Code repository for the KDD 2025 paper **"Advancing Graph Foundation Models: A Data-Centric Perspective"**.

**DCGFM** is a plug-and-play, data-centric approach for Graph Foundation Models (GFM) using data pruning. A model-agnostic **hard pruning** module filters out uninformative subgraphs; a model-aware **soft pruning** module dynamically filters low-contribution subgraphs each training epoch.

![DCGFM Framework](images/framework.jpg)

---

## Requirements

| Component | Version |
|---|---|
| GPU | RTX 5090 (sm_120 / Blackwell) or older |
| CUDA | 12.8 |
| Python | 3.10 |
| PyTorch | 2.12.0 nightly+cu128 (**required** for RTX 5090) |

> **Note for older GPUs (sm_80/86/90):** Use stable PyTorch 2.6+cu124 and install PyG from [pyg.org wheels](https://data.pyg.org/whl/) instead of building from source.

---

## Quick Start (Docker — Recommended)

### 1. Build the image

```bash
docker build -t dcgfm .
```

> First build takes ~20–30 min (compiles PyG extensions from source for sm_120).

### 2. Start the container

```bash
docker run --gpus all -it --name dcgfm_modern \
  -v $(pwd):/workspace/DCGFM \
  dcgfm
```

### 3. Or reuse the existing container

```bash
docker start dcgfm_modern
docker exec -it dcgfm_modern /bin/bash
```

---

## Manual Environment Setup (without Docker)

```bash
# 1. PyTorch nightly (RTX 5090 / sm_120)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 2. Remaining packages
pip install -r requirements.txt

# 3. Build PyG extensions from source
python docker/build_pyg.py

# 4. Patch OGB for PyTorch 2.6+ compatibility
python docker/patch_ogb.py
```

---

## OFA with DCGFM

### Data Preparation

**Sentence-BERT model** — download automatically at first run, or manually from [HuggingFace](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1) and place in `OFA/cache_data/model/`.

**Molecule dataset** — download from [HuggingFace](https://huggingface.co/datasets/haitengzhao/molecule_property_instruction) and place extracted files in `OFA/cache_data/dataset/molecule_property_instruction/`.

All other datasets (ogbn-arxiv, FB15K237, etc.) are **downloaded automatically** on first run.

### Run Pre-training + Evaluation

```bash
cd OFA/
python run_cdm.py \
  --control_gpu --gpus 0 --save_model \
  --override yamls/soft_and_hard.yaml \
  --hard_pruning_mode hard_prune_api \
  --hard_pruning_joint \
  --hard_pruning_reverse \
  --hard_pruning_ratio 0.5 \
  --prune_ratio 0.5
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--hard_pruning_ratio` | `0.0` | Fraction of subgraphs removed by hard pruning (0.3 / 0.5 / 0.7) |
| `--prune_ratio` | `0.5` | Soft pruning ratio per epoch |
| `--hard_pruning_mode` | — | `hard_prune_api` uses anomaly scores via API |
| `--hard_pruning_joint` | flag | Apply pruning jointly across tasks |
| `--hard_pruning_reverse` | flag | Keep low-scoring (hard) samples |
| `--num_epochs` | `30` | Training epochs (set in yaml) |
| `--checkpoint_interval` | `10` | Save checkpoint every N epochs |

### Expected Outputs

```
OFA/
├── lightning_logs/version_X/checkpoints/
│   ├── OFA_soft_and_hard_..._epoch=9.ckpt
│   ├── OFA_soft_and_hard_..._epoch=19.ckpt
│   ├── OFA_soft_and_hard_..._epoch=29.ckpt
│   └── last.ckpt
├── saved_exp/<timestamp>/          ← WandB offline logs
└── cache_data/ofa_dataset/         ← Preprocessed graph + text embeddings
```

Final test results (mean ± std over 10 runs) are printed to stdout for:
- `arxiv_fs` — few-shot node classification on ogbn-arxiv
- `FB15K237_fs` — few-shot link prediction
- `mol_fs` — few-shot molecular property prediction

### Monitor Training

```bash
# Inside the container, if running in background:
tail -f /tmp/ofa_run.log
```

---

## GraphCLIP with DCGFM

### Data Preparation

Download pre-training datasets and place in `GraphCLIP/summary/`:

| Dataset | Link |
|---|---|
| OGBN-ArXiv | [Google Drive](https://drive.google.com/file/d/1AeAnnqPui05FuBX7JvWQMJA8kr2CIFYS/view) |
| ArXiv\_2023 | [Google Drive](https://drive.google.com/file/d/1t1icJvRtw9OBpc88uws_wIsKFoVHtM0D/view) |
| Reddit | [Google Drive](https://drive.google.com/file/d/1c7gtoy918suLlUN5a8CYUGCEbzYAeSeX/view) |
| OGBN-Products | [Google Drive](https://drive.google.com/file/d/1IAmU8mAJ-rVzFu1iOkvQes1RtS8-RU-M/view) |

Download target datasets and place in `GraphCLIP/processed_data/`:

| Dataset | Link |
|---|---|
| WikiCS | [Google Drive](https://drive.google.com/file/d/1vOo_Iql19Eccgr8t6H70AYIvxwu87846/view) |
| Instagram | [Google Drive](https://drive.google.com/file/d/1c9ZkdHyDHKaInGnmXlLGjYIPeTY-njF7/view) |
| Ele-Photo | [Google Drive](https://drive.google.com/file/d/1qFMixgszCODpo7e7syhucUjKYr75T8cx/view) |
| Ele-Computers | [Google Drive](https://drive.google.com/file/d/1487we3C9AJryvAMCCH0W7YA0nXFQ1H8o/view) |
| Books-History | [Google Drive](https://drive.google.com/file/d/1zAlK6BdQy0YmwPu9M5GXbImLrDQS4BON/view) |

Generate target subgraphs:
```bash
cd GraphCLIP/
bash gen_target_subg.sh
```

### Step 1 — Hard Pruning

```bash
cd GraphCLIP/
python hard_pruning.py \
  --source_data ogbn-arxiv+arxiv_2023+pubmed+ogbn-products+reddit \
  --threshold 30
```

Replace `30` with `50` or `70` for other pruning ratios.

### Step 2 — Training with Soft Pruning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --source_data ogbn-arxiv+arxiv_2023+pubmed+ogbn-products+reddit \
  --batch_size 7200 \
  --epochs 30
```

### Step 3 — Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --target_data cora+citeseer+wikics+instagram+photo+computer+history \
  --ckpt graphclip
```

---

## Known Issues & Fixes Applied

| Issue | Fix |
|---|---|
| `torch.load` default `weights_only=True` in PyTorch ≥2.6 breaks PyG/OGB deserialization | `docker/patch_ogb.py` patches all OGB files; `weights_only=False` added to OFA data loaders |
| RTX 5090 (sm_120) not supported by stable PyTorch | Use PyTorch 2.12 nightly+cu128 |
| PyG wheels not available for PyTorch nightly | Build from source via `docker/build_pyg.py` |
| Sentence-BERT HuggingFace cache path mismatch | Auto-resolved via snapshot directory lookup in `OFA/models/model.py` |

---

## Citation

```bibtex
@inproceedings{dcgfm2025,
  title={Advancing Graph Foundation Models: A Data-Centric Perspective},
  booktitle={KDD},
  year={2025}
}
```



16 may 2026

experiment prunes the farther ones 70 % form hard pruning

F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe run_cdm.py `
    --override yamls/soft_and_hard_mol_far.yaml `
    --hard_pruning_mode hard_prune_api `
    --hard_pruning_ratio 0.7 `
    --prune_ratio 0.3 `
    --hard_pruning_epochs 25 `
    --offline_log




Current architecture (for context)

Input SMILES graph
      ↓
llm_proj (ST embeddings → 768d)   ← BinGraphAttModel
      ↓
PyGRGCNEdge × 5 layers             ← MultiLayerMessagePassing
  each layer: RGCNEdgeConv (node + edge features)
      ↓
JK = "none" → returns h_list[0..4]  ← list of all 5 layer outputs
      ↓
SingleHeadAtt pools across layers   ← BinGraphAttModel.att
      ↓
MLP → prediction
The BinGraphAttModel with JK="none" is already doing learned layer aggregation (attention over all 5 layer outputs). That's good — it's already a form of JK. But the receptive field is still hard-capped at 5 hops.

Suggested changes for deeper GIN
Change 1 — Add residual connections (prevents over-smoothing)
In GNN.py, MultiLayerMessagePassing.forward() currently:


h = self.layer_forward(layer, message)   # no skip
Change to:


h_new = self.layer_forward(layer, message)
h = h_new + message["h"] if layer > 0 else h_new   # residual from layer 1 onward
Since inp_dim == out_dim == 768 throughout, dimensions already match. Without this, stacking 10+ layers causes all node embeddings to converge to the same value (over-smoothing) — residuals are what make deep GNNs work.

Change 2 — Increase depth to 10 layers (config-only)
In yamls/soft_and_hard_mol_far.yaml:


num_layers: 10   # was 5
dropout: 0.1     # was 0.0 — deeper nets need regularisation
With residuals from Change 1, 10 layers covers the median diameter (13) at 77% — up from 1.4% coverage today.

Change 3 — Virtual node (biggest single win, requires new class)
The MultiLayerMessagePassingVN already exists in GNN.py (lines 126–196). It adds a learnable virtual node that aggregates all node embeddings every layer and broadcasts back — effectively shortcutting the diameter to 2 for any pair of atoms regardless of actual graph distance.

This requires creating PyGRGCNEdgeVN that inherits from MultiLayerMessagePassingVN instead of MultiLayerMessagePassing, and swapping the instantiation in run_cdm.py.

Priority order
Change	Impact	Effort	Risk
1. Residual connections	High — enables going deep safely	~5 lines in GNN.py	Low
2. num_layers: 10	Medium — doubles receptive field	Config only	Low (needs Change 1 first)
3. Virtual node	Highest — fixes long-range entirely	New class in model.py	Medium




adaptive gin archite cture
$env:WANDB_MODE="disabled"; F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe F:\Rabia-Salman\DCGFM\OFA\run_cdm_adaptive.py --override yamls/soft_and_hard_mol_adaptive.yaml --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 --prune_ratio 0.3 --hard_pruning_epochs 25



virtual node
$env:WANDB_MODE="disabled"; F:\Rabia-Salman\DCGFM\dcgfm\Scripts\python.exe F:\Rabia-Salman\DCGFM\OFA\run_cdm_vn.py --override yamls/soft_and_hard_mol_vn.yaml --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 --prune_ratio 0.3 --hard_pruning_epochs 25




define architecture of gin layers
Read F:\Rabia-Salman\DCGFM\OFA\gp\nn\layer\pyg.py
The model is actually not standard GIN — it's RGCNEdgeConv, a relational GCN with edge features. Here's the complete architecture from input to output:

Layer-by-layer architecture
Input to each layer
Each node v has:

Node embedding h_v ∈ R^768 (projected from SentenceTransformer)
Edge attributes e_{uv} ∈ R^768 (also projected)
Edge type r_{uv} ∈ {0,1,2,3,4} (5 relation types — hence num_rels=5)
Single RGCNEdgeConv layer (gp/nn/layer/pyg.py:16)
Step 1 — Message (line 70):


m_{uv} = ReLU(h_u + e_{uv})     ← neighbour node + edge feature fused
Step 2 — Aggregate by relation type (lines 54–59):


For each relation r in {0..4}:
    agg_r(v) = mean( m_{uv}  for all u where edge_type(u→v) == r )
    out(v)  += agg_r(v) @ W_r       ← relation-specific weight matrix W_r ∈ R^{768×768}
Step 3 — Self-loop + bias (lines 61–63):


out(v) += h_v @ W_root + b
Final output per layer: h_v^{(l)} = BatchNorm( out(v) ), then ReLU (except last layer).

Each layer has 5 weight matrices W_r (one per relation) + W_root + b → 5×768×768 + 768×768 + 768 = 3,539,712 params/layer.

Full stack (PyGRGCNEdge / PyGRGCNEdgeDeep)

Input graph (batch of molecules)
    │
    ▼
Layer 0:  RGCNEdgeConv(768→768)  →  BN  →  ReLU
    │  [+ residual if Deep variant]
Layer 1:  RGCNEdgeConv(768→768)  →  BN  →  ReLU
    │
    ⋮
Layer N-1: RGCNEdgeConv(768→768)  →  BN  (no ReLU on last)
    │
    ▼
JK="none" → returns [h^0, h^1, …, h^{N-1}]   ← list of all layer outputs
Prediction head (BinGraphAttModel)

h_list = [h^0 … h^{N-1}]           shape: N tensors of [num_nodes, 768]
    │
    ▼
stack → [num_nodes, N_layers, 768]
    │
SingleHeadAtt(query=h_input, key=stack, value=stack)
    │                                 ← learns WHICH layer is most useful per node
    ▼
h_out ∈ R^768  per node
    │
select true_nodes_mask
    │
MLP(768 → 1536 → 768 → 1)
    │
logits → BCEWithLogitsLoss
What makes this NOT standard GIN
Standard GIN	This model (RGCN-Edge)
Aggregation	Sum	Mean, per relation type
Edge features	Ignored	Fused into message (h_u + e_{uv})
Relation types	None	5 separate weight matrices
Expressiveness	WL-test equivalent	Handles heterogeneous edges
The 5 relation types encode different bond/graph-structural roles (bond type, distance, etc.), which is why it's better suited for multi-domain tasks (node, link, mol) than a plain GIN.