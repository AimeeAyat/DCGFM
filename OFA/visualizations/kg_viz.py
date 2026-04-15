"""
Knowledge Graph visualization — FB15K237 + WN18RR
  Visual 1 — FB15K237: graph sample + top-30 relation distribution
  Visual 2 — WN18RR: graph sample + all 11 relations
  Visual 3 — KG triplet structure + NOI for e2e_link and lr_link tasks
  Visual 4 — Real entity/relation text examples for both datasets
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

import json, random
random.seed(42); np.random.seed(42)

print("=== FB15K237 + WN18RR ===")

KG_PATH = os.path.join(OFA, "data/KG")

# ── load FB15K237 ─────────────────────────────────────────────────────────────
fb_triplets = []
with open(os.path.join(KG_PATH, "FB15K237/train.txt")) as f:
    for line in f:
        h, r, t = line.strip().split()
        fb_triplets.append((h, r, t))

with open(os.path.join(KG_PATH, "FB15K237/entity2wikidata.json")) as f:
    fb_entities = json.load(f)

fb_relations = [r for _, r, _ in fb_triplets]
from collections import Counter
fb_rel_count = Counter(fb_relations)

print(f"  FB15K237 — triplets: {len(fb_triplets)}, unique relations: {len(fb_rel_count)}")

# ── load WN18RR ───────────────────────────────────────────────────────────────
wn_triplets = []
with open(os.path.join(KG_PATH, "WN18RR/train.txt")) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            wn_triplets.append(tuple(parts))

wn_entity2text = {}
with open(os.path.join(KG_PATH, "WN18RR/entity2text.txt")) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            wn_entity2text[parts[0]] = parts[1]

wn_relations = [r for _, r, _ in wn_triplets]
wn_rel_count  = Counter(wn_relations)

print(f"  WN18RR   — triplets: {len(wn_triplets)}, unique relations: {len(wn_rel_count)}")

# ── Visual 1: FB15K237 graph sample + relation distribution ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("FB15K237 — Freebase Knowledge Graph  |  14,541 entities, 237 relations",
             fontsize=13, fontweight="bold")

# 1a: graph sample
ax = axes[0]
n_sample = 300
sample_t  = random.sample(fb_triplets, n_sample)
G = nx.DiGraph()
for h, r, t in sample_t:
    G.add_edge(h, t, relation=r)
G.remove_edges_from(nx.selfloop_edges(G))
pos = nx.spring_layout(G, seed=42, k=0.6)
# color edges by relation
rels_in_g  = list({d["relation"] for _, _, d in G.edges(data=True)})
rel2color  = {r: plt.cm.tab20(i / max(len(rels_in_g), 1)) for i, r in enumerate(rels_in_g)}
edge_colors = [rel2color[G[u][v]["relation"]] for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#FF8A65",
                       node_shape="s", node_size=30, alpha=0.8)
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                       width=0.6, alpha=0.5, arrows=True, arrowsize=5)
ax.set_title(f"{n_sample} triplet sample\nSquares = Freebase entities\nEdge color = relation type", fontsize=9)
ax.axis("off")

# 1b: top 30 relations
ax = axes[1]
top30 = fb_rel_count.most_common(30)
labels_r, counts_r = zip(*top30)
short_labels = [l.split("/")[-1][:20] for l in labels_r]
ax.barh(short_labels, counts_r, color="#FF8A65", edgecolor="white")
ax.set_xlabel("# triplets", fontsize=8)
ax.set_title("Top 30 Freebase relations\n(237 total)", fontsize=9)
ax.tick_params(axis='y', labelsize=6.5)
ax.invert_yaxis()

# 1c: entity text examples
ax = axes[2]; ax.axis("off"); ax.set_facecolor("#FFF3E0")
ex = "FB15K237 ENTITY TEXT EXAMPLES:\n\n"
shown = 0
for eid, info in list(fb_entities.items())[:15]:
    if shown >= 8: break
    label = info.get("label","?")
    desc  = info.get("description","")[:60] if info.get("description") else "N/A"
    alts  = ", ".join(info.get("alternatives",[])[:2])
    ex   += f"Entity: {label}\n  desc: {desc}\n  alts: {alts}\n\n"
    shown += 1
ax.text(0.02, 0.98, ex, transform=ax.transAxes, fontsize=6.5,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("Entity feature text\n(name + description → Sentence-BERT)", fontsize=9)

save(fig, "kg_1_fb15k237.png")

# ── Visual 2: WN18RR graph sample + all 11 relations ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("WN18RR — WordNet Knowledge Graph  |  40,943 synsets, 11 lexical relations",
             fontsize=13, fontweight="bold")

# 2a: graph sample
ax = axes[0]
sample_wn  = random.sample(wn_triplets, min(200, len(wn_triplets)))
G_wn = nx.DiGraph()
for h, r, t in sample_wn:
    G_wn.add_edge(h, t, relation=r)
G_wn.remove_edges_from(nx.selfloop_edges(G_wn))
rel_list_wn = list(wn_rel_count.keys())
r2c_wn = {r: plt.cm.tab20(i / max(len(rel_list_wn), 1)) for i, r in enumerate(rel_list_wn)}
ec_wn  = [r2c_wn[G_wn[u][v]["relation"]] for u, v in G_wn.edges()]
pos_wn = nx.spring_layout(G_wn, seed=42, k=0.8)
nx.draw_networkx_nodes(G_wn, pos_wn, ax=ax, node_color="#E57373",
                       node_shape="s", node_size=30, alpha=0.8)
nx.draw_networkx_edges(G_wn, pos_wn, ax=ax, edge_color=ec_wn,
                       width=0.8, alpha=0.6, arrows=True, arrowsize=6)
ax.set_title("200 triplet sample\nSquares = WordNet synsets\nEdge color = relation type", fontsize=9)
ax.axis("off")

# 2b: all 11 relations
ax = axes[1]
rel_labels_wn = [k.replace("_", " ") for k in wn_rel_count.keys()]
rel_counts_wn = list(wn_rel_count.values())
colors11 = plt.cm.tab20(np.linspace(0, 1, len(rel_counts_wn)))
bars = ax.barh(rel_labels_wn, rel_counts_wn, color=colors11, edgecolor="white")
ax.set_xlabel("# triplets in train set", fontsize=8)
ax.set_title("ALL 11 WN18RR relations", fontsize=9)
ax.tick_params(axis='y', labelsize=8)
for bar, c in zip(bars, rel_counts_wn):
    ax.text(c + 50, bar.get_y() + bar.get_height()/2,
            str(c), va="center", fontsize=7)
ax.invert_yaxis()

# 2c: entity text examples
ax = axes[2]; ax.axis("off"); ax.set_facecolor("#FCE4EC")
sample_ents = random.sample(list(wn_entity2text.items()), 10)
ex = "WN18RR ENTITY TEXT EXAMPLES:\n\n"
for eid, desc in sample_ents:
    ex += f"Synset: {eid}\n  '{desc[:70]}'\n\n"
ex += "\nRelation types are lexical:\nhypernym, hyponym, meronym,\nsimilar_to, also_see,\nverb_group, domain_topic..."
ax.text(0.02, 0.98, ex, transform=ax.transAxes, fontsize=6.5,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("Synset feature text\n(synset name + definition → Sentence-BERT)", fontsize=9)

save(fig, "kg_2_wn18rr.png")

# ── Visual 3: KG triplet + NOI construction ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("KG Tasks — How Triplets Become Prompt Subgraphs with NOI",
             fontsize=12, fontweight="bold")

# pick a real FB15K237 triplet
h_ex, r_ex, t_ex = fb_triplets[42]
h_label = fb_entities.get(h_ex, {}).get("label", h_ex)[:15]
t_label = fb_entities.get(t_ex, {}).get("label", t_ex)[:15]
rel_short = r_ex.split("/")[-1][:20]

# panel 1: raw triplet
ax = axes[0]; ax.axis("off")
ax.set_facecolor("#FFF8E1")
G_t = nx.DiGraph()
G_t.add_edge("HEAD\n"+h_label, "TAIL\n"+t_label, label=rel_short)
pos_t = {"HEAD\n"+h_label: (0, 0), "TAIL\n"+t_label: (2, 0)}
nx.draw_networkx(G_t, pos_t, ax=ax, node_color="#FF8A65", node_shape="s",
                 node_size=2000, font_size=7, font_weight="bold", arrows=True)
nx.draw_networkx_edge_labels(G_t, pos_t, edge_labels={("HEAD\n"+h_label,"TAIL\n"+t_label): rel_short},
                              ax=ax, font_size=7)
ax.set_title("A KG Triplet (head, relation, tail)\nThe atomic unit of a knowledge graph", fontsize=9)
ax.text(0.5, 0.1, f"FB15K237 example:\n({h_label}, {rel_short}, {t_label})",
        ha="center", transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round", fc="white"))

# panel 2: e2e_link — predict relation type
feat_nodes = list(range(8))
noi_id     = 100
class_ids  = [200 + i for i in range(5)]   # 5 sample relation types
top5_rels  = [k.split("/")[-1][:12] for k, _ in fb_rel_count.most_common(5)]
sub_edges  = [(i, i+1) for i in range(7)]
draw_noi_graph(axes[1], feat_nodes, noi_id, class_ids, sub_edges,
    title="e2e_link\nRelation Type Prediction",
    task_desc="2-hop subgraph around (head,tail)\nNOI attached to all entities\n237 relation type class nodes\nQ: What relation connects them?",
    class_labels=top5_rels + ["..."])

# panel 3: lr_link — few-shot relation prediction
draw_noi_graph(axes[2], feat_nodes[:6], noi_id, [], sub_edges[:5],
    title="lr_link\nFew-Shot Relation Prediction",
    task_desc="Predict relation with few examples.\nNo class nodes in subgraph.\nSupport triplets compared by GNN.\nSource/Target nodes get special\nprompt edges [1,2] vs [3,4].")

save(fig, "kg_3_noi_construction.png")

# ── Visual 4: relation type taxonomy ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("KG Relation Taxonomy — What the 237 + 11 Relations Represent",
             fontsize=12, fontweight="bold")

# FB15K237 relation categories
ax = axes[0]
fb_domains = Counter()
for r in fb_rel_count:
    parts = r.strip("/").split("/")
    domain = parts[1] if len(parts) > 1 else "other"
    fb_domains[domain] += fb_rel_count[r]
top_domains = fb_domains.most_common(20)
domain_labels, domain_counts = zip(*top_domains)
y_pos = list(range(len(domain_labels)))
ax.barh(y_pos, domain_counts, color="#FF8A65", edgecolor="white")
ax.set_yticks(y_pos); ax.set_yticklabels(domain_labels, fontsize=7.5)
ax.set_xlabel("# triplets", fontsize=8)
ax.invert_yaxis()
ax.set_title("FB15K237: Top 20 Freebase domains\n(e.g. /people, /film, /music, /sports)", fontsize=9)

# WN18RR relation explanation
ax = axes[1]; ax.axis("off"); ax.set_facecolor("#FCE4EC")
wn_desc = {
    "_hypernym":      "A is a type of B\n  e.g. 'dog' hypernym 'animal'",
    "_hyponym":       "B is a type of A (inverse hypernym)\n  e.g. 'poodle' hyponym 'dog'",
    "_member_meronym":"A is a member of B\n  e.g. 'player' meronym 'team'",
    "_part_meronym":  "A is a part of B\n  e.g. 'wheel' meronym 'car'",
    "_substance_meronym": "A is substance of B",
    "_also_see":      "A and B are related concepts",
    "_verb_group":    "verbs in same group",
    "_similar_to":    "similar adjectives",
    "_member_holonym":"B is a member of A",
    "_part_holonym":  "B is part of A",
    "_domain_topic_region": "A is in domain B",
}
ex = "WN18RR ALL 11 RELATIONS EXPLAINED:\n\n"
for rel, desc in wn_desc.items():
    cnt = wn_rel_count.get(rel, 0)
    ex += f"{rel} ({cnt:,} triplets)\n  {desc}\n\n"
ax.text(0.02, 0.98, ex, transform=ax.transAxes, fontsize=7,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax.set_title("WN18RR: All 11 WordNet relations explained", fontsize=9)

save(fig, "kg_4_relation_taxonomy.png")
print("KG done.\n")
