"""
model_adaptive.py — Adaptive-depth GIN for molecular graphs.

Architecture:
  • One 20-layer residual GIN (PyGRGCNEdgeDeep, JK="none") — single forward pass.
  • Per-graph diameter is computed via scipy BFS on the fly.
  • Each molecule's nodes attend ONLY over the layer-subset matching its diameter:
        diameter ≤  5  →  attend layers  1-5   (default depth)
        diameter  6-10  →  attend layers  1-10
        diameter > 10   →  attend layers  1-20
  • Masked single-head attention replaces the uniform BinGraphAttModel attention.
  • Shared MLP head for classification.

This is a drop-in replacement for BinGraphAttModel.
Used by run_cdm_adaptive.py (original run_cdm.py is untouched).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gp.nn.models.util_model import MLP
from models.model import LLM_DIM_DICT
from models.model_deep import PyGRGCNEdgeDeep


# ── Diameter → depth mapping ─────────────────────────────────────────────────
THRESH_SMALL  = 5    # diameter ≤ 5  → use first 5 layers
THRESH_MEDIUM = 10   # diameter ≤ 10 → use first 10 layers
                     # diameter > 10 → use all 20 layers

DEPTH_SMALL   = 5
DEPTH_MEDIUM  = 10
DEPTH_LARGE   = 20   # total GIN layers


# ── Masked attention ─────────────────────────────────────────────────────────

class MaskedSingleHeadAtt(nn.Module):
    """Single-head attention with an optional boolean key mask.

    mask shape: [batch, n_keys] — True = attend, False = ignore (set to -inf).
    """

    def __init__(self, dim):
        super().__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(dim, dtype=torch.float))
        self.Wk = nn.Parameter(torch.zeros(dim, dim))
        self.Wq = nn.Parameter(torch.zeros(dim, dim))
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wq)

    def forward(self, key, query, value, mask=None):
        # key/value: [N, n_keys, dim]  query: [N, 1, dim]
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim  # [N, 1, n_keys]
        if mask is not None:
            score = score.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn  = F.softmax(score, dim=-1)
        ctx   = torch.bmm(attn, value)   # [N, 1, dim]
        return ctx, attn


# ── Diameter computation ─────────────────────────────────────────────────────

def _compute_batch_diameters(g):
    """Return per-graph diameter tensor [n_graphs] computed via scipy BFS.

    For molecular graphs (< ~100 atoms) this takes a few microseconds per
    graph and is negligible compared to GNN forward pass time.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path

    n_graphs   = g.num_graphs
    edge_index = g.edge_index.cpu()
    batch_cpu  = g.batch.cpu()
    src_all, dst_all = edge_index[0], edge_index[1]

    diameters = []
    for gid in range(n_graphs):
        node_mask   = batch_cpu == gid
        global_nodes = node_mask.nonzero(as_tuple=True)[0]
        n_nodes     = global_nodes.shape[0]

        if n_nodes <= 1:
            diameters.append(0)
            continue

        # Remap global → local node indices
        local_map          = torch.empty(batch_cpu.shape[0], dtype=torch.long)
        local_map[global_nodes] = torch.arange(n_nodes)

        edge_mask  = node_mask[src_all]
        lsrc = local_map[src_all[edge_mask]].numpy()
        ldst = local_map[dst_all[edge_mask]].numpy()

        if len(lsrc) == 0:
            diameters.append(0)
            continue

        adj   = sp.csr_matrix(
            (np.ones(len(lsrc)), (lsrc, ldst)), shape=(n_nodes, n_nodes)
        )
        dists  = shortest_path(adj, directed=False)
        finite = dists[np.isfinite(dists)]
        diameters.append(int(finite.max()) if len(finite) > 0 else 0)

    return torch.tensor(diameters, dtype=torch.long, device=g.x.device)


def _diameter_to_depth(diam: int) -> int:
    if diam <= THRESH_SMALL:
        return DEPTH_SMALL
    elif diam <= THRESH_MEDIUM:
        return DEPTH_MEDIUM
    return DEPTH_LARGE


# ── Main model ───────────────────────────────────────────────────────────────

class AdaptiveGINModel(nn.Module):
    """Adaptive-depth GIN: one 20-layer residual GIN + diameter-gated attention.

    Accepts the same constructor signature as BinGraphAttModel so it can be
    patched into run_cdm.py without touching the original file.
    The `model` argument (external GNN) is intentionally ignored — this class
    builds its own internal GNN.
    """

    def __init__(
        self,
        model=None,          # ignored — we build our own GNN
        llm_name: str = "ST",
        outdim:   int = 768,
        task_dim: int = 1,
        add_rwpe=None,       # not used (can be added later)
        dropout:  float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.outdim    = outdim
        self.llm_proj  = nn.Linear(LLM_DIM_DICT[llm_name], outdim)

        # Single 20-layer residual GIN — JK="none" returns all layer outputs
        self.gnn = PyGRGCNEdgeDeep(
            num_layers = DEPTH_LARGE,
            num_rels   = 5,
            inp_dim    = outdim,
            out_dim    = outdim,
            drop_ratio = dropout,
            JK         = "none",
        )

        self.att = MaskedSingleHeadAtt(outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)

    def forward(self, g):
        # 1. Project LLM node / edge features to GNN dimension
        g.x         = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)

        # 2. Single 20-layer forward pass → list of 20 node-embedding tensors
        h_list = self.gnn(g)                          # list[20] of [N_nodes, dim]
        emb    = torch.stack(h_list, dim=1)           # [N_nodes, 20, dim]

        # 3. Per-graph diameter → allowed depth
        diameters  = _compute_batch_diameters(g)      # [n_graphs]
        depths     = torch.tensor(
            [_diameter_to_depth(d.item()) for d in diameters],
            dtype=torch.long, device=g.x.device,
        )                                              # [n_graphs]
        node_depth = depths[g.batch]                  # [N_nodes]

        # 4. Attention mask: node i attends to layer l only if l ≤ node_depth[i]
        layer_idx  = torch.arange(1, DEPTH_LARGE + 1, device=g.x.device)  # [20]
        attn_mask  = layer_idx.unsqueeze(0) <= node_depth.unsqueeze(1)    # [N_nodes, 20]

        # 5. Masked attention pools across the allowed layers per node
        query  = g.x.unsqueeze(1)                     # [N_nodes, 1, dim]
        h_out  = self.att(emb, query, emb, mask=attn_mask)[0].squeeze(1)  # [N_nodes, dim]

        # 6. Classify target nodes
        class_emb = h_out[g.true_nodes_mask]
        return self.mlp(class_emb)

    def freeze_gnn_parameters(self):
        for p in self.gnn.parameters():
            p.requires_grad = False
        for p in self.att.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False
