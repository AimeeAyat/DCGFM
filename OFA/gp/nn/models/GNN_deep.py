"""
GNN_deep.py — Residual-enabled message-passing base class.

Drop-in replacement for MultiLayerMessagePassing when using deep GNNs
(num_layers > 6).  Adds a per-layer residual skip connection starting at
layer 1 to prevent over-smoothing.

Usage:
    Subclass DeepMultiLayerMessagePassing instead of MultiLayerMessagePassing.
    Everything else (build_input_layer, build_hidden_layer, layer_forward, …)
    stays identical to the original.
"""

import torch
import torch.nn.functional as F

from gp.nn.models.GNN import MultiLayerMessagePassing


class DeepMultiLayerMessagePassing(MultiLayerMessagePassing):
    """MultiLayerMessagePassing + residual skip connections.

    From layer 1 onward: h = layer_output + previous_h
    Layer 0 is kept as-is (input projection, no residual).

    Works without modification because inp_dim == out_dim for all layers
    in the RGCN-Edge setup used in this project.
    """

    def forward(self, g, drop_mask=None):
        h_list = []
        prev_h = None

        message = self.build_message_from_input(g)

        for layer in range(self.num_layers):
            h = self.layer_forward(layer, message)

            if self.batch_norm:
                h = self.batch_norm[layer](h)

            # Residual: skip connection from the previous layer's output.
            # Skipped at layer 0 because there is no previous output yet.
            if prev_h is not None:
                h = h + prev_h

            if layer != self.num_layers - 1:
                h = F.relu(h)

            if self.drop_ratio is not None:
                dropped_h = F.dropout(h, p=self.drop_ratio, training=self.training)
                if drop_mask is not None:
                    keep = torch.logical_not(drop_mask).view(-1, 1)
                    drop = drop_mask.view(-1, 1)
                    h = drop * dropped_h + keep * h
                else:
                    h = dropped_h

            prev_h = h
            message = self.build_message_from_output(g, h)
            h_list.append(h)

        if self.JK == "last":
            return h_list[-1]
        elif self.JK == "sum":
            out = h_list[0]
            for layer in range(1, self.num_layers):
                out = out + h_list[layer]
            return out
        elif self.JK == "mean":
            out = h_list[0]
            for layer in range(1, self.num_layers):
                out = out + h_list[layer]
            return out / self.num_layers
        else:
            return h_list  # JK="none" — BinGraphAttModel handles pooling across layers
