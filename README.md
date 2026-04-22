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
