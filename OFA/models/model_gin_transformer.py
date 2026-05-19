"""
model_gin_transformer.py — Design A: Fixed 10-layer residual GIN + 1 Transformer block.

No routing. No diameter computation. Every molecule goes through the same pipeline:

    LLM projection (768d)
    → 10-layer PyGRGCNEdgeDeep  (residual RGCN-Edge, local chemistry)
    → 1 Transformer block        (8-head self-attention, pre-norm, global context)
    → true_nodes_mask selection
    → MLP → logits

Rationale:
  - BiScale-GTR (2026) empirically confirmed: shallow GNN + Transformer > deep GNN alone.
  - Fixed at 10 layers because the 15-layer experiment showed degradation vs 10-layer
    (chemhiv 0.485 vs 0.576) — the over-smoothing ceiling is already visible at 15L.
  - Transformer self-attention connects all atoms in one step regardless of diameter,
    solving the long-range problem without requiring more GIN layers.
  - Single Transformer block keeps parameter count manageable.

Drop-in replacement for BinGraphAttModel.
Used by run_cdm_gin_transformer.py.
"""

import torch
import torch.nn as nn

from gp.nn.models.util_model import MLP
from models.model import LLM_DIM_DICT
from models.model_deep import PyGRGCNEdgeDeep


# ── Padding helpers ───────────────────────────────────────────────────────────

def _pack(h, batch):
    """
    Pack flat [num_nodes, dim] into padded [B, max_N, dim] + bool key-padding mask.

    mask[b, i] = True  means position i is padding (MultiheadAttention ignores it).
    """
    n_graphs = int(batch.max().item()) + 1
    counts   = batch.bincount(minlength=n_graphs)   # [B]
    max_n    = int(counts.max().item())
    D        = h.size(-1)

    h_pad  = h.new_zeros(n_graphs, max_n, D)
    mask   = torch.ones(n_graphs, max_n, dtype=torch.bool, device=h.device)

    offset = 0
    for i, cnt in enumerate(counts.tolist()):
        h_pad[i, :cnt] = h[offset : offset + cnt]
        mask[i, :cnt]  = False          # real atoms, not padding
        offset += cnt

    return h_pad, mask, counts


def _unpack(h_pad, counts):
    """Unpack [B, max_N, dim] back to flat [num_nodes, dim]."""
    return torch.cat([h_pad[i, :cnt] for i, cnt in enumerate(counts.tolist())], dim=0)


# ── Transformer block (pre-norm) ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block.
        h = h + Dropout(Attn(LayerNorm(h)))
        h = h + Dropout(FFN(LayerNorm(h)))
    Applied per-graph via padded batching.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, h, batch):
        """
        h     : [num_nodes, dim]
        batch : [num_nodes]  (graph index per node)
        returns [num_nodes, dim]
        """
        h_pad, pad_mask, counts = _pack(h, batch)   # [B, max_N, dim]

        # Self-attention with padding mask
        x     = self.norm1(h_pad)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        h_pad = h_pad + self.drop(attn_out)

        # Feed-forward
        h_pad = h_pad + self.drop(self.ffn(self.norm2(h_pad)))

        return _unpack(h_pad, counts)               # [num_nodes, dim]


# ── Main model ────────────────────────────────────────────────────────────────

class GINTransformerModel(nn.Module):
    """
    Fixed 10-layer residual GIN + 1 Transformer block for ALL molecules.

    Accepts the same constructor signature as BinGraphAttModel so it can be
    patched into run_cdm.py without touching the original file.
    The external `model` arg (pre-built GNN) is intentionally ignored.
    """

    GIN_LAYERS = 10

    def __init__(
        self,
        model    = None,        # ignored — we build our own
        llm_name : str   = "ST",
        outdim   : int   = 768,
        task_dim : int   = 1,
        add_rwpe         = None,   # not used
        dropout  : float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)

        # 10-layer residual GIN — JK="last" returns a single tensor
        self.gnn = PyGRGCNEdgeDeep(
            num_layers = self.GIN_LAYERS,
            num_rels   = 5,
            inp_dim    = outdim,
            out_dim    = outdim,
            drop_ratio = dropout,
            JK         = "last",
        )

        self.transformer = TransformerBlock(outdim, num_heads=8, dropout=dropout)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)

    def forward(self, g):
        # 1. Project LLM features
        g.x         = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)

        # 2. Local chemistry via residual GIN
        h = self.gnn(g)                        # [num_nodes, dim]

        # 3. Global context via Transformer (all atoms attend to each other)
        h = self.transformer(h, g.batch)       # [num_nodes, dim]

        # 4. Classify target nodes
        class_emb = h[g.true_nodes_mask]
        return self.mlp(class_emb)

    def freeze_gnn_parameters(self):
        for p in self.gnn.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False
