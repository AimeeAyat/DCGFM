"""
model_deep.py — PyGRGCNEdgeDeep: residual RGCN-Edge GNN.

Drop-in replacement for PyGRGCNEdge from models/model.py.
Identical in every way except it inherits from DeepMultiLayerMessagePassing
(residual skip connections) instead of MultiLayerMessagePassing.

Used by run_cdm_deep.py.
"""

from gp.nn.layer.pyg import RGCNEdgeConv
from gp.nn.models.GNN_deep import DeepMultiLayerMessagePassing


class PyGRGCNEdgeDeep(DeepMultiLayerMessagePassing):
    """RGCN-Edge with residual connections for deeper molecular GNNs."""

    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm)
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, g):
        return {"g": g.edge_index, "h": g.x, "e": g.edge_type, "he": g.edge_attr}

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](
            message["h"], message["he"], message["g"], message["e"]
        )