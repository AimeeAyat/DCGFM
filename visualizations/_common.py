"""
Shared utilities for all gen_data visualizations.
"""
import os, sys, torch

ROOT = os.path.dirname(os.path.dirname(__file__))   # DCGFM/
OFA  = os.path.join(ROOT, "OFA")
sys.path.insert(0, OFA)

# patch torch.load for PyTorch >= 2.6
_orig = torch.load
def _load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _load

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

OUT = os.path.join(ROOT, "visualizations", "output")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved -> {path}")
    plt.close(fig)

# ── NOI subgraph builder ──────────────────────────────────────────────────────
def build_noi_subgraph(edge_index, num_nodes, target_node, hop=2, max_nodes=30):
    """Extract k-hop subgraph around target_node, return as networkx graph."""
    adj = {i: set() for i in range(num_nodes)}
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].add(d); adj[d].add(s)
    visited = {target_node}
    frontier = {target_node}
    for _ in range(hop):
        nxt = set()
        for n in frontier:
            nxt |= adj[n]
        frontier = nxt - visited
        visited |= frontier
        if len(visited) >= max_nodes:
            break
    nodes = list(visited)[:max_nodes]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for s, d in zip(src, dst):
        if s in visited and d in visited:
            G.add_edge(s, d)
    return G, nodes

def draw_noi_graph(ax, feature_nodes, noi_id, class_nodes, edges,
                   title, task_desc, class_labels=None):
    """
    Draw a prompt-augmented subgraph showing:
      - feature nodes (papers/entities)
      - NOI node (task prompt)
      - class nodes (prediction targets)
      - prompt edges between them
    """
    G = nx.DiGraph()
    G.add_nodes_from(feature_nodes)
    G.add_node(noi_id)
    G.add_nodes_from(class_nodes)
    for s, d in edges:
        G.add_edge(s, d)

    # connect NOI to feature nodes
    for fn in feature_nodes:
        G.add_edge(fn, noi_id)
        G.add_edge(noi_id, fn)
    # connect NOI to class nodes
    for cn in class_nodes:
        G.add_edge(noi_id, cn)
        G.add_edge(cn, noi_id)

    pos = nx.spring_layout(G, seed=1, k=2.5)
    # place NOI in center
    pos[noi_id] = np.array([0.0, 0.0])
    # spread class nodes at top
    for i, cn in enumerate(class_nodes):
        angle = np.pi * (i / max(len(class_nodes)-1, 1))
        pos[cn] = np.array([np.cos(angle)*1.5, np.sin(angle)*1.5 + 0.5])

    # draw layers
    nx.draw_networkx_nodes(G, pos, nodelist=feature_nodes, ax=ax,
                           node_color="#64B5F6", node_size=120, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=[noi_id], ax=ax,
                           node_color="#FF7043", node_size=350,
                           node_shape="D", alpha=1.0)
    nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, ax=ax,
                           node_color="#A5D6A7", node_size=220,
                           node_shape="s", alpha=0.9)
    nx.draw_networkx_edges(G, pos,
                           edgelist=[(s,d) for s,d in edges
                                     if s in feature_nodes and d in feature_nodes],
                           ax=ax, edge_color="#aaa", width=0.8, alpha=0.5,
                           arrows=False)
    prompt_edges = ([(fn, noi_id) for fn in feature_nodes] +
                    [(noi_id, fn) for fn in feature_nodes] +
                    [(noi_id, cn) for cn in class_nodes] +
                    [(cn, noi_id) for cn in class_nodes])
    nx.draw_networkx_edges(G, pos, edgelist=prompt_edges, ax=ax,
                           edge_color="#FF7043", width=1.2,
                           style="dashed", alpha=0.7, arrows=True, arrowsize=8)

    labels = {noi_id: "NOI"}
    if class_labels:
        for cn, lbl in zip(class_nodes, class_labels):
            labels[cn] = lbl[:12]
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6)

    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.text(0.01, 0.01, task_desc, transform=ax.transAxes,
            fontsize=6, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    legend = [
        mpatches.Patch(color="#64B5F6", label="Feature node"),
        mpatches.Patch(color="#FF7043", label="NOI node"),
        mpatches.Patch(color="#A5D6A7", label="Class node"),
    ]
    ax.legend(handles=legend, fontsize=5.5, loc="lower right")
    ax.axis("off")
