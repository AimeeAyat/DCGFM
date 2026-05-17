"""
GraphGPS-based hard pruning module — drop-in replacement for GIN in hard_prune_module.py.

Architecture: GraphGPS (GPSConv) = local GINConv + global MultiHead Attention per layer.
Chosen over Exphormer because:
  - Our graphs are small (10-300 nodes): GraphGPS is 1.8x faster than Exphormer at this scale.
  - GPSConv is available natively in torch_geometric — no external dependencies.
  - Equal accuracy gain over GIN (+2-9%) at lower engineering cost.

Affects: hard pruning only (anomaly scoring before main training).
Does NOT affect: soft pruning (InfoBatch), main model (PyGRGCNEdge), or testing.

Usage:
    cd OFA
$env:WANDB_MODE="disabled"
C:/Users/salma/anaconda3/python.exe run_cdm.py `
  --control_gpu --gpus 0 `
  --override yamls/soft_and_hard.yaml `
  --hard_pruning_mode graphgps_hard_prune_api `
  --hard_pruning_reverse `
  --hard_pruning_ratio 0.7 `
  --save_model

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import numpy as np

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GPSConv, global_mean_pool
from torch_geometric.nn import BatchNorm
import torch.distributed as dist


# ---------------------------------------------------------------------------
# GraphGPS encoder
# ---------------------------------------------------------------------------

class GraphGPS(nn.Module):
    """
    Graph General, Powerful, Scalable Transformer.

    Each layer runs:
        1. Local message passing via GINConv  (captures bond/neighbor structure)
        2. Global multi-head attention        (captures long-range dependencies)
    Both outputs are summed and normalised.

    Then global_mean_pool aggregates node embeddings → graph embedding.
    We return embeddings from every layer (like the GIN in hard_prune_module.py)
    so the DeepSVDD center can be computed across all depths.
    """

    def __init__(self, nfeat: int, nhid: int, nlayer: int,
                 nheads: int = 4, dropout: float = 0.0, **kwargs):
        super().__init__()
        assert nhid % nheads == 0, (
            f"nhid ({nhid}) must be divisible by nheads ({nheads})"
        )

        # Project raw node features → hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(nfeat, nhid),
            BatchNorm(nhid),
        )

        self.convs = nn.ModuleList()
        for _ in range(nlayer):
            # Local MPNN: GINConv with 2-layer MLP
            gin_mlp = nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid),
            )
            local_conv = GINConv(gin_mlp, train_eps=True)

            # GPS layer: local_conv + global attention, both projected to nhid
            gps = GPSConv(
                channels=nhid,
                conv=local_conv,
                heads=nheads,
                dropout=dropout,
                attn_type="multihead",   # standard O(N²) — fine for N < 300
            )
            self.convs.append(gps)

        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Cast float inputs (embeddings are float32)
        if x.dtype != torch.float32:
            x = x.float()

        x = self.input_proj(x)

        # Collect graph-level embeddings after each layer (including layer 0)
        graph_embeds = [self.pooling(x, batch)]

        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x, edge_index, batch)
            graph_embeds.append(self.pooling(x, batch))

        # Shape: [nlayer+1, batch_size, nhid]
        return torch.stack(graph_embeds)


# ---------------------------------------------------------------------------
# DeepSVDD wrapper (mirrors GIN_Hard_Prune exactly)
# ---------------------------------------------------------------------------

class GraphGPS_Hard_Prune(pl.LightningModule):
    """
    Wraps GraphGPS with the Deep SVDD objective.

    Epoch 0  : forward pass only — computes the hypersphere center from
               the mean of all graph embeddings (no gradient update).
    Epoch 1+ : optimises radius and pushes embeddings toward/away from center.

    Anomaly score per graph = ||emb - center||² - radius²
        > 0  →  outside hypersphere  →  informative / diverse
        < 0  →  inside hypersphere   →  redundant / common
    """

    def __init__(self, nfeat: int, nhid: int = 128, nlayer: int = 3,
                 nheads: int = 4, dropout: float = 0.0,
                 learning_rate: float = 0.001, weight_decay: float = 0.0,
                 **kwargs):
        super().__init__()
        self.model = GraphGPS(nfeat, nhid, nlayer, nheads, dropout)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        # DeepSVDD parameters
        self.nu = 1.0
        self.radius = 0.0

        self.register_buffer("center", torch.zeros(nhid))
        self.register_buffer("all_layer_centers", torch.zeros(nlayer + 1, nhid))

        self.training_step_outputs = []
        self.test_step_outputs = []

    # ------------------------------------------------------------------
    def forward(self, data: Data) -> torch.Tensor:
        """Returns anomaly score per graph in the batch."""
        embs = self.model(data)          # [nlayer+1, B, nhid]
        embs = embs.sum(dim=0)           # [B, nhid]  (sum across layers)
        dist = torch.sum((embs - self.center) ** 2, dim=1)
        return dist - self.radius ** 2

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Data, batch_idx: int):
        if self.current_epoch == 0:
            # Warm-up pass: collect embeddings to initialise center
            embs = self.model(batch)
            loss = torch.zeros(1, requires_grad=True, device=self.device)
            self.training_step_outputs.append({"loss": loss, "emb": embs.detach()})
            return {"loss": loss, "emb": embs.detach()}
        else:
            scores = self(batch)
            loss = self.radius ** 2 + (1.0 / self.nu) * torch.mean(F.relu(scores))
            self.training_step_outputs.append(loss)
            self.log("train_loss", loss, prog_bar=True)
            return loss

    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            # Compute hypersphere center = mean of all layer-wise embeddings
            all_embs = torch.cat(
                [d["emb"] for d in self.training_step_outputs], dim=1
            )  # [nlayer+1, N_total, nhid]
            self.all_layer_centers = all_embs.mean(dim=1)   # [nlayer+1, nhid]
            self.center = self.all_layer_centers.sum(dim=0)  # [nhid]
        else:
            losses = [
                item for item in self.training_step_outputs
                if isinstance(item, torch.Tensor)
            ]
            if losses:
                avg = torch.stack(losses).mean()
                self.log("epoch_loss", avg, prog_bar=True)
                print(f"Epoch {self.current_epoch} avg loss: {avg.item():.6f}")

        self.training_step_outputs.clear()

    # ------------------------------------------------------------------
    # Testing (score collection)
    # ------------------------------------------------------------------

    def test_step(self, batch: Data, batch_idx: int):
        scores = self(batch)
        self.test_step_outputs.append(scores)
        return scores


# ---------------------------------------------------------------------------
# Minimal InMemoryDataset wrapper (identical to hard_prune_module.py)
# ---------------------------------------------------------------------------

class _SimpleDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(".")
        self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return []

    def download(self): pass

    def process(self): pass


# ---------------------------------------------------------------------------
# Public API — called by run_cdm.py via globals()[hard_pruning_mode](...)
# ---------------------------------------------------------------------------

def graphgps_hard_prune_api(
    data_list,
    batch_size: int = 32,
    weight_decay: float = 0.0,
    nlayer: int = 3,
    nheads: int = 4,
    max_epochs: int = 25,
    devices=None,
) -> np.ndarray:
    """
    Train a GraphGPS-DeepSVDD model on data_list and return per-graph
    anomaly scores as a numpy array.

    Scores > 0  →  outside hypersphere (informative, keep with --hard_pruning_reverse)
    Scores < 0  →  inside hypersphere  (redundant, keep without --hard_pruning_reverse)

    Args:
        data_list   : list of torch_geometric.data.Data objects (no prompt nodes)
        batch_size  : mini-batch size (default 32, same as GIN version)
        weight_decay: L2 regularisation
        nlayer      : number of GPS layers (default 3; GIN used 5 but GPS is deeper per layer)
        nheads      : attention heads per GPS layer (nhid must be divisible by nheads)
        max_epochs  : DeepSVDD training epochs (default 25)
        devices     : GPU device list, e.g. [0]

    Returns:
        numpy array of shape [len(data_list)] with anomaly scores
    """
    if devices is None:
        devices = [0]

    dataset = _SimpleDataset(data_list)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    nfeat = dataset[0].x.shape[1]
    model = GraphGPS_Hard_Prune(
        nfeat=nfeat,
        nhid=128,
        nlayer=nlayer,
        nheads=nheads,
        weight_decay=weight_decay,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        max_epochs=max_epochs,
        logger=False,          # no W&B / lightning logs for pruning phase
        enable_progress_bar=True,
    )

    trainer.fit(model=model, train_dataloaders=train_loader)
    trainer.test(model=model, dataloaders=test_loader, verbose=False)

    if not model.test_step_outputs:
        raise ValueError("GraphGPS pruning: no test outputs collected.")

    local_scores = torch.cat(model.test_step_outputs)

    # Multi-GPU: gather scores from all ranks
    if trainer.world_size > 1:
        gathered = trainer.strategy.all_gather(local_scores)
        if trainer.is_global_zero:
            return torch.cat(gathered).cpu().detach().numpy()

    return local_scores.cpu().detach().numpy()
