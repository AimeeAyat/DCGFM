"""
model_vn.py — PyGRGCNEdgeVN: RGCN-Edge with a correctly-wired virtual node.

Drop-in replacement for PyGRGCNEdge.  Used by run_cdm_vn.py.
"""

from gp.nn.layer.pyg import RGCNEdgeConv
from gp.nn.models.GNN_vn import VNMultiLayerMessagePassing


class PyGRGCNEdgeVN(VNMultiLayerMessagePassing):
    """RGCN-Edge + virtual node (broadcast-correct implementation)."""

    def __init__(
        self,
        num_layers: int,
        num_rels:   int,
        inp_dim:    int,
        out_dim:    int,
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
