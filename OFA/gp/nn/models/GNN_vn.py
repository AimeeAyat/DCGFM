"""
GNN_vn.py — Correct Virtual Node implementation.

The existing MultiLayerMessagePassingVN in GNN.py updates the virtual node
embedding each layer but never broadcasts it back to the node features —
making it a no-op on message passing.

This file fixes that.  Before every layer l > 0 each node embedding receives
the virtual node embedding of its graph:

    h_v  ←  h_v  +  vnode[graph(v)]        (broadcast)
    h_v'  ←  RGCN-layer(h_v, edge_index)   (message passing)
    vnode  ←  MLP(sum_v(h_v') + vnode)     (aggregate + update)

Reference: OGB virtual-node baseline
  https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import scatter

from gp.nn.models.GNN import MultiLayerMessagePassing
from gp.nn.models.util_model import MLP


class VNMultiLayerMessagePassing(MultiLayerMessagePassing):
    """MultiLayerMessagePassing with a correctly-wired virtual node.

    Virtual node lifecycle per forward pass
    ───────────────────────────────────────
    Init:    vnode  = Embedding(1, dim)  initialised to 0   [n_graphs, dim]

    Layer 0: (no broadcast yet — node features are freshly projected)
             h'  = conv_0( h )
             vnode = MLP_0( sum_nodes(h') + vnode )

    Layer l (l > 0):
             h   = h  +  vnode[graph(node)]     ← broadcast
             h'  = conv_l( h )
             vnode = MLP_l( sum_nodes(h') + vnode )   (except last layer)

    JK:  "last" | "sum" | "none"  (all supported)
    """

    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        drop_ratio=None,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm)

        self.virtualnode_embedding = nn.Embedding(1, self.out_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # One MLP per inter-layer gap  (num_layers - 1 gaps)
        self.virtualnode_mlp_list = nn.ModuleList(
            MLP([self.out_dim, 2 * self.out_dim, self.out_dim])
            for _ in range(self.num_layers - 1)
        )

    def forward(self, g):
        h_list = []

        message = self.build_message_from_input(g)

        # g.batch maps every node to its graph index — already [n_nodes]
        batch_seg = g.batch
        n_graphs  = int(batch_seg.max().item()) + 1

        # Per-graph virtual node — one vector per graph, broadcast to nodes
        vnode = self.virtualnode_embedding(
            torch.zeros(n_graphs, dtype=torch.int, device=g.x.device)
        )  # [n_graphs, dim]

        for layer in range(self.num_layers):

            # ── Broadcast vnode → nodes (every layer except the first) ──────
            if layer > 0:
                h_boosted = message["h"] + vnode[batch_seg]
                message = {**message, "h": h_boosted}

            # ── Message passing ──────────────────────────────────────────────
            h = self.layer_forward(layer, message)
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.relu(h)
            if self.drop_ratio is not None:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)

            message = self.build_message_from_output(g, h)
            h_list.append(h)

            # ── Update virtual node (all layers except the last) ─────────────
            if layer < self.num_layers - 1:
                vnode_agg  = scatter(h, batch_seg, dim=0, dim_size=n_graphs)
                vnode_temp = vnode_agg + vnode
                vnode = F.dropout(
                    self.virtualnode_mlp_list[layer](vnode_temp),
                    self.drop_ratio if self.drop_ratio else 0,
                    training=self.training,
                )

        if self.JK == "last":
            return h_list[-1]
        elif self.JK == "sum":
            out = h_list[0]
            for l in range(1, self.num_layers):
                out = out + h_list[l]
            return out
        else:  # "none" — BinGraphAttModel handles layer pooling
            return h_list
