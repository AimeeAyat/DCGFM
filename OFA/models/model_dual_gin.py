"""
model_dual_gin.py — Dual VN-GIN with diameter-based hard split.

Two 10-layer VN-GIN experts, each responsible for half the diameter range:
    GIN 1  (small specialist) : diameter <= THRESHOLD  (~first  50% of molecules)
    GIN 2  (large specialist) : diameter >  THRESHOLD  (~second 50% of molecules)

THRESHOLD is set to 13 (median of chemblpre, chempcba) and is also used as the
split point for the two independent Deep-SVDD hard-pruning passes — so each GIN
is trained on data pruned relative to its OWN structural centre, not a global one.

Routing is HARD (not soft / learned): no router MLP, no load-balancing loss.
Each molecule routes to exactly one GIN based on its graph diameter.

Drop-in replacement for BinGraphAttModel.
Used by run_cdm_dual.py.
"""

import torch
import torch.nn as nn

from gp.nn.models.util_model import MLP
from models.model import LLM_DIM_DICT
from models.model_deep import PyGRGCNEdgeDeep
from models.model_adaptive import _compute_batch_diameters


DIAMETER_THRESHOLD = 13   # default fallback; overridden by run_cdm_dual at runtime


class DualVNGINModel(nn.Module):
    """
    Two 10-layer VN-GINs with hard diameter routing.

    Each molecule is processed by exactly one expert:
        diameter <= threshold  →  gin_small  (first  50% by diameter order)
        diameter >  threshold  →  gin_large  (second 50% by diameter order)

    The threshold is set by run_cdm_dual.py after computing the actual 50/50
    boundary from the training data — call set_diameter_threshold() before training.

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
        self._threshold = DIAMETER_THRESHOLD   # set after super().__init__()
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)

        # GIN 1 — small-diameter specialist (diameter ≤ 13)
        self.gin_small = PyGRGCNEdgeDeep(
            num_layers=10, num_rels=5, inp_dim=outdim, out_dim=outdim,
            drop_ratio=dropout, JK="last",
        )
        # GIN 2 — large-diameter specialist (diameter > 13)
        self.gin_large = PyGRGCNEdgeDeep(
            num_layers=10, num_rels=5, inp_dim=outdim, out_dim=outdim,
            drop_ratio=dropout, JK="last",
        )

        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)

    def set_diameter_threshold(self, threshold: int):
        """Called by run_cdm_dual after computing the data-driven 50/50 boundary."""
        self._threshold = threshold
        print(f"DualVNGINModel: diameter threshold set to {threshold}")

    def forward(self, g):
        # 1. Shared LLM projection
        g.x         = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)

        # 2. Per-graph diameter → node-level routing mask
        diameters  = _compute_batch_diameters(g)                    # [n_graphs]
        node_small = (diameters <= self._threshold)[g.batch]        # [n_nodes] bool

        # 3. Run both GINs on the full batch
        h1 = self.gin_small(g)    # [n_nodes, dim]
        h2 = self.gin_large(g)    # [n_nodes, dim]

        # 4. Hard select: each node takes output from its designated GIN
        h = torch.where(node_small.unsqueeze(-1), h1, h2)

        # 5. Classify
        return self.mlp(h[g.true_nodes_mask])

    def freeze_gnn_parameters(self):
        for mod in [self.gin_small, self.gin_large, self.mlp, self.llm_proj]:
            for p in mod.parameters():
                p.requires_grad = False
