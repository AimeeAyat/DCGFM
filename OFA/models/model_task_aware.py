"""
model_task_aware.py — Task-type-aware hybrid model.

Architecture
============
  Non-mol graphs (arxiv, FB15K237, WN18RR, ...):
      5-layer RGCN-Edge + layer-attention head (mirrors the original BinGraphAttModel)

  Molecular graphs (chemblpre → chemhiv/chempcba):
      Small MoE — 3 experts (E1=3L RGCN, E2=3L RGCN, E3=3L GPS),
      sparse top-2 routing on 4-descriptor topology vector,
      load-balancing auxiliary loss (Switch Transformer, λ=0.005)

Routing
=======
  Each episode graph carries  g.is_mol = tensor([True/False])  (shape [1]).
  After PyG batching:  g.is_mol  has shape [n_graphs].
  Expanded to node level:  node_is_mol = g.is_mol[g.batch]  (shape [n_nodes]).

  torch.where(node_is_mol, h_mol, h_base)  routes each node's embedding
  to the correct sub-model and propagates gradients only through the
  selected path.

Cache safety
============
  The is_mol flag is added at __getitem__ time (fresh graph construction).
  It does NOT affect:
    - all_no_prompt_data  (only used for SVDD, not passed to the model)
    - episode index cache  (class_ind_*.pkl etc. — just indices)
    - hard pruning cache   (dcgfm_hard_prune_api_25_0.7_*.pkl — reused as-is)

Drop-in replacement for BinGraphAttModel.
Used by run_cdm_task_aware.py.
"""

import torch
import torch.nn as nn

from torch_geometric.nn import GPSConv
from torch_geometric.nn.conv import GINEConv

from gp.nn.models.util_model import MLP
from models.model import LLM_DIM_DICT, PyGRGCNEdge, SingleHeadAtt
from models.model_deep import PyGRGCNEdgeDeep
from models.model_moe import GPSExpert, _topo_features, _sparse_top2, _load_balance_loss


AUX_WEIGHT = 0.005


class TaskAwareHybridModel(nn.Module):
    """
    Task-type-aware hybrid:
        node/link graphs  →  5L RGCN-Edge + layer attention  (original baseline)
        mol graphs        →  3-expert MoE (RGCN + RGCN + GPS)

    Accepts the same constructor signature as BinGraphAttModel.
    External `model` argument is ignored.
    """

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

        # ── Non-mol path: original 5L RGCN-Edge + layer attention ────────────
        # Mirrors BinGraphAttModel(PyGRGCNEdge(5L, JK="none"))
        self.gin_base = PyGRGCNEdge(
            num_layers=5, num_rels=5, inp_dim=outdim, out_dim=outdim,
            drop_ratio=dropout, JK="none",          # returns list of 5 tensors
        )
        self.base_att = SingleHeadAtt(outdim)        # learns which layer to trust

        # ── Mol path: Small MoE (3L×3 experts) ───────────────────────────────
        self.e1 = PyGRGCNEdgeDeep(
            num_layers=3, num_rels=5, inp_dim=outdim, out_dim=outdim,
            drop_ratio=dropout, JK="last",
        )
        self.e2 = PyGRGCNEdgeDeep(
            num_layers=3, num_rels=5, inp_dim=outdim, out_dim=outdim,
            drop_ratio=dropout, JK="last",
        )
        self.e3 = GPSExpert(outdim, heads=8, dropout=dropout)   # 3L GPS

        self.router = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3),
        )

        # ── Shared classification head ────────────────────────────────────────
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)

        self.last_aux_loss = None

    # ── Sub-model forwards ────────────────────────────────────────────────────

    def _base_forward(self, g):
        """5L RGCN-Edge with layer-attention pooling (mirrors BinGraphAttModel)."""
        h_list = self.gin_base(g)                          # list of 5 tensors [n_nodes, dim]
        emb    = torch.stack(h_list, dim=1)                # [n_nodes, 5, dim]
        query  = g.x.unsqueeze(1)                          # [n_nodes, 1, dim]
        h      = self.base_att(emb, query, emb)[0].squeeze(1)  # [n_nodes, dim]
        return h

    def _mol_forward(self, g):
        """3-expert MoE with topology-aware sparse top-2 routing."""
        topo    = _topo_features(g)                        # [n_graphs, 4]
        logits  = self.router(topo)                        # [n_graphs, 3]
        weights = _sparse_top2(logits)                     # [n_graphs, 3] sparse

        if self.training:
            # Load-balance only on mol graphs that are actually in this batch
            self.last_aux_loss = _load_balance_loss(weights)
        else:
            self.last_aux_loss = None

        h1 = self.e1(g)
        h2 = self.e2(g)
        h3 = self.e3(g.x, g.edge_index, g.batch, g.edge_attr, g.edge_type)

        w  = weights[g.batch]                              # [n_nodes, 3]
        return w[:, 0:1] * h1 + w[:, 1:2] * h2 + w[:, 2:3] * h3

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(self, g):
        # 1. Shared LLM projection
        g.x         = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)

        # 2. Run both paths
        h_base = self._base_forward(g)      # [n_nodes, dim]
        h_mol  = self._mol_forward(g)       # [n_nodes, dim]

        # 3. Route per node: torch.where propagates gradients only through
        #    the selected path, so each sub-model only trains on its task type.
        if hasattr(g, 'is_mol') and g.is_mol is not None:
            node_is_mol = g.is_mol[g.batch].bool()         # [n_nodes]
            h = torch.where(node_is_mol.unsqueeze(-1), h_mol, h_base)
        else:
            h = h_base   # fallback if is_mol flag missing

        return self.mlp(h[g.true_nodes_mask])

    def freeze_gnn_parameters(self):
        for mod in [self.gin_base, self.base_att,
                    self.e1, self.e2, self.e3, self.router,
                    self.mlp, self.llm_proj]:
            for p in mod.parameters():
                p.requires_grad = False
