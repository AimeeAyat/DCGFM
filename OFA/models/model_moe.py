"""
model_moe.py — Topology-Aware Graph Mixture of Experts (GraphMoE).

Architecture
============
Three architecture-diverse experts, sparse top-2 routing on a 4-descriptor
topological feature vector computed per graph:

    s = [ diameter/20,          # range of molecular structure
          n_atoms/50,           # molecular size
          cyclomatic/10,        # ring-count proxy  (E - N + 1)
          mean_degree/4 ]       # atom connectivity density

Experts:
    E1 — 5-layer  residual RGCN-Edge  (local chemistry, ring/fragment specialist)
    E2 — 10-layer residual RGCN-Edge  (medium-range chain specialist)
    E3 — 5-layer  GPS                 (parallel GINEConv + MultiheadAttention,
                                       long-range SOTA per NeurIPS 2022 / GPS++)

GPS layer (E3) per depth l:
    h_local  = GINEConv(h, edge_index, edge_attr)    ← local bond-aware MPNN
    h_global = MultiheadAttention(h)                  ← global all-atom attention
    h        = LayerNorm( MLP(h_local + h_global) )  ← parallel fusion

Why GPS > sequential GIN → Transformer:
  - GPS paper (NeurIPS 2022) ablation: "major drop when using only Transformer;
    MPNN is essential." Sequential runs Transformer after GIN is done — GPS runs
    both IN PARALLEL at every depth, refining local+global simultaneously.
  - OGB molhiv: GPS = 0.788 vs GIN+VN = 0.774.
  - Our own exp: sequential GIN+Transformer = 0.526 (worse than 5L baseline).

Routing (top-2 sparse, graph-level expanded to node-level):
    logits  = RouterMLP(s)               [n_graphs, 3]
    weights = sparse_top2(logits)        [n_graphs, 3]
    h       = Σ_i weights[:,i]*Expert_i  [n_nodes,  dim]

Load-balancing aux loss (Switch Transformer, λ=0.01):
    L_aux = M * Σ_i  f_i * p_i  stored in self.last_aux_loss

Drop-in replacement for BinGraphAttModel.
Used by run_cdm_moe.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GPSConv
from torch_geometric.nn.conv import GINEConv

from gp.nn.models.util_model import MLP
from models.model import LLM_DIM_DICT
from models.model_deep import PyGRGCNEdgeDeep
from models.model_adaptive import _compute_batch_diameters


# ── GPS Expert (E3) ───────────────────────────────────────────────────────────

class GPSExpert(nn.Module):
    """
    3-layer GPS stack: each layer = GINEConv (local, edge-aware) +
    MultiheadAttention (global) run IN PARALLEL, outputs summed.

    Edge-type fix: E1/E2 (RGCNEdge) condition on discrete bond relation types
    (edge_type ∈ {0..4}).  GINEConv ignores edge_type and only sees edge_attr.
    We close this gap by adding a learned edge-type embedding to edge_attr
    before every GPS layer, so E3 also respects bond relation types.
    """

    N_LAYERS   = 3
    NUM_RELS   = 5      # same as RGCN-Edge (5 bond relation types)

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Learned embedding per bond relation type, added to edge_attr
        self.edge_type_emb = nn.Embedding(self.NUM_RELS, dim)

        self.layers = nn.ModuleList()
        for _ in range(self.N_LAYERS):
            local_conv = GINEConv(
                nn=nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * dim, dim),
                ),
                train_eps=True,
                edge_dim=dim,
            )
            self.layers.append(GPSConv(
                channels   = dim,
                conv       = local_conv,
                heads      = heads,
                dropout    = dropout,
                attn_type  = "multihead",
                norm       = "batch_norm",
            ))

    def forward(self, x, edge_index, batch, edge_attr, edge_type):
        # Fuse LLM edge embeddings with discrete bond-type embeddings
        edge_attr = edge_attr + self.edge_type_emb(edge_type)
        for layer in self.layers:
            x = layer(x, edge_index, batch, edge_attr=edge_attr)
        return x                    # [n_nodes, dim]


# ── Topological feature computation ──────────────────────────────────────────

def _topo_features(g):
    """4 normalised topological descriptors per graph → [n_graphs, 4] on device."""
    batch     = g.batch
    n_graphs  = int(batch.max().item()) + 1
    batch_cpu = batch.cpu()
    ei_cpu    = g.edge_index.cpu()

    n_atoms    = batch_cpu.bincount(minlength=n_graphs).float()
    edge_batch = batch_cpu[ei_cpu[0]]
    n_edges    = edge_batch.bincount(minlength=n_graphs).float() / 2.0
    mean_deg   = n_edges * 2.0 / n_atoms.clamp(min=1)
    cyclomatic = (n_edges - n_atoms + 1).clamp(min=0)
    diameters  = _compute_batch_diameters(g).float().cpu()   # BFS, ~1 ms/batch

    features = torch.stack([
        (diameters  / 20.0).clamp(0, 1),
        (n_atoms    / 50.0).clamp(0, 1),
        (cyclomatic / 10.0).clamp(0, 1),
        (mean_deg   /  4.0).clamp(0, 1),
    ], dim=1)                               # all on CPU

    return features.to(g.x.device)


# ── Sparse top-2 routing ──────────────────────────────────────────────────────

def _sparse_top2(logits):
    top2_vals, top2_idx = torch.topk(logits, k=2, dim=-1)
    weights = torch.zeros_like(logits)
    weights.scatter_(1, top2_idx, F.softmax(top2_vals, dim=-1))
    return weights


# ── Load-balancing auxiliary loss ─────────────────────────────────────────────

def _load_balance_loss(weights):
    M = weights.shape[1]
    f = (weights > 0).float().mean(dim=0)
    p = weights.mean(dim=0)
    return M * (f * p).sum()


# ── Main model ────────────────────────────────────────────────────────────────

class GraphMoEModel(nn.Module):
    """
    Topology-Aware Graph MoE — three architecture-diverse experts,
    sparse top-2 routing, Switch Transformer load-balancing loss.
    Drop-in replacement for BinGraphAttModel.
    """

    AUX_WEIGHT = 0.005   # halved: lighter load-balancing pressure on small experts

    def __init__(
        self,
        model    = None,
        llm_name : str   = "ST",
        outdim   : int   = 768,
        task_dim : int   = 1,
        add_rwpe         = None,
        dropout  : float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)

        # E1: local chemistry — 3-layer residual RGCN-Edge
        self.e1 = PyGRGCNEdgeDeep(
            num_layers=3, num_rels=5, inp_dim=outdim, out_dim=outdim,
            drop_ratio=dropout, JK="last",
        )
        # E2: medium range — 3-layer residual RGCN-Edge
        self.e2 = PyGRGCNEdgeDeep(
            num_layers=3, num_rels=5, inp_dim=outdim, out_dim=outdim,
            drop_ratio=dropout, JK="last",
        )
        # E3: long range — 3-layer GPS (GINEConv + Attention parallel per layer)
        self.e3 = GPSExpert(outdim, heads=8, dropout=dropout)

        # Router: 4 topological features → 3 expert logits
        self.router = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
        self.last_aux_loss = None

    def forward(self, g):
        # 1. Shared LLM projection
        g.x         = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)

        # 2. Topology-aware routing
        topo    = _topo_features(g)
        logits  = self.router(topo)
        weights = _sparse_top2(logits)            # [n_graphs, 3]

        # 3. Load-balancing loss (training only)
        self.last_aux_loss = _load_balance_loss(weights) if self.training else None

        # 4. Run all three experts
        h1 = self.e1(g)                                              # [n_nodes, dim]
        h2 = self.e2(g)                                              # [n_nodes, dim]
        h3 = self.e3(g.x, g.edge_index, g.batch, g.edge_attr, g.edge_type)  # [n_nodes, dim]

        # 5. Node-level weighted sum
        w = weights[g.batch]                                         # [n_nodes, 3]
        h = w[:, 0:1] * h1 + w[:, 1:2] * h2 + w[:, 2:3] * h3

        return self.mlp(h[g.true_nodes_mask])

    def freeze_gnn_parameters(self):
        for mod in [self.e1, self.e2, self.e3, self.router, self.mlp, self.llm_proj]:
            for p in mod.parameters():
                p.requires_grad = False
