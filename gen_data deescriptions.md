Full data flow

pubmed.pt
   │
   ├─ 19,717 nodes (papers)
   ├─ raw_texts → Sentence-BERT → 768-dim node embeddings
   ├─ edge_index → citation connections
   └─ y → 3 class labels (Diabetes Exp / Type1 / Type2)
            │
            ▼
   OFAPygDataset.process()
            │
            ▼
   Subgraphs extracted per training node
   + prompt nodes/edges attached
            │
            ▼
   Input to hard pruning (GIN Deep SVDD)

Purpose
Loads the Pubmed citation graph and prepares all text descriptions that will be encoded by Sentence-BERT into node/edge features for the OFA pipeline.

get_data(dset) — the only function
Step 1 — Load the raw graph

data = torch.load("pubmed.pt")
Loads a pre-saved PyG Data object containing:

data.num_nodes — 19,717 paper nodes
data.edge_index — citation edges
data.raw_texts — list of paper title+abstract strings (one per node)
data.y — class label per node (0, 1, or 2)

nx_g = pyg.utils.to_networkx(data, to_undirected=True)
edge_index = torch.tensor(list(nx_g.edges())).T
Converts to NetworkX and back — this deduplicates edges and ensures the graph is truly undirected (no duplicate directed pairs).

Step 2 — Build text descriptions
This is the core of the file. Everything below becomes input to Sentence-BERT.


clean_text = ["feature node. paper title and abstract: " + t for t in text]
One string per paper node — what the node "is".


label_text = ["prompt node. literature category and description: " + desc
              for desc in ordered_desc]   # 3 entries: Exp, Type1, Type2
One string per class — what each Pubmed category means.


edge_label_text = [
    "prompt node. two papers do not have co-citation",
    "prompt node. two papers have co-citation",
]
Used for link prediction tasks — binary labels for whether two papers are co-cited.


edge_text = ["feature edge. connected papers are cited together by other papers."]
One shared description for all edges.


noi_node_text   = ["prompt node. node classification on the paper's category"]
noi_node_edge_text = ["prompt node. link prediction on the papers that are cited together"]
NOI = Node Of Interest — a special prompt node added to each subgraph that tells the model what task it is doing.


prompt_edge_text = ["prompt edge", 
                    "prompt edge. edge for query graph that is our target",
                    "prompt edge. edge for support graph that is an example"]
Edges connecting prompt/NOI nodes to the subgraph — used in few-shot task construction.

Step 3 — Return everything

return (
    [new_data],           # the full Pubmed graph
    [clean_text, edge_text, noi_node_text + noi_node_edge_text,
     label_text + edge_label_text, prompt_edge_text],   # all text lists
    task_map              # which text goes with which task
)

The task_map maps task names to which text features to use:

Task	What it does
e2e_node	End-to-end node classification (which of 3 Diabetes categories?)
lr_node	Few-shot node classification (low-resource, needs support examples)
e2e_link	End-to-end link prediction (are these two papers co-cited?)


File hierarchy

data/
├── single_graph/
│   ├── gen_data.py          ← router / base class
│   ├── arxiv/gen_data.py    ← ogbn-arxiv loader
│   ├── Cora/gen_data.py     ← Cora loader
│   ├── Pubmed/gen_data.py   ← Pubmed loader
│   └── wikics/gen_data.py   ← WikiCS loader
├── KG/gen_data.py           ← FB15K237 + WN18RR loader
└── chemmol/gen_data.py      ← molecule loader
1. single_graph/gen_data.py — the router

AVAILABLE_DATA = ["Cora", "Pubmed", "wikics", "arxiv"]

class SingleGraphOFADataset(OFAPygDataset):
    def gen_data(self):
        data_module = importlib.import_module("data.single_graph." + self.name + ".gen_data")
        return data_module.get_data(self)
Purpose: A single class that handles ALL 4 citation/social datasets. Uses importlib to dynamically load the right gen_data.py based on the dataset name. You never call Cora/Pubmed/arxiv/WikiCS loaders directly — they're all routed through this.


def add_text_emb(self, data_list, text_emb):
    data_list[0].node_text_feat    = text_emb[0]   # paper embeddings
    data_list[0].edge_text_feat    = text_emb[1]   # edge description embedding
    data_list[0].noi_node_text_feat = text_emb[2]  # task prompt embedding
    data_list[0].class_node_text_feat = text_emb[3] # class label embeddings
    data_list[0].prompt_edge_text_feat = text_emb[4] # prompt edge embeddings
After Sentence-BERT encodes the raw texts, this attaches all 5 embedding matrices to the PyG Data object.

get_edge_list() defines how subgraphs are assembled per task — which edges connect feature nodes → NOI node → class nodes:

Mode	Edges built
e2e_node	f→n, n→f, n→c, c→n (full prompt graph)
lr_node	f→n, n→f only (few-shot, no class nodes in graph)
e2e_link	f→n, n→f, n→c, c→n
2. arxiv/gen_data.py — ogbn-arxiv
Graph: 169,343 paper nodes, 1.17M directed citation edges, 40 CS arXiv categories.


pyg_data = PygNodePropPredDataset(name="ogbn-arxiv", root=dset.data_dir)
Downloads ogbn-arxiv via OGB if not cached.


feat_node_texts = get_node_feature(cur_path)   # title + abstract per paper
class_node_texts = get_label_feature(cur_path) # 40 category descriptions
logic_node_texts = get_logic_feature(cur_path) # "not X and not Y" / "either X or Y"
Three types of node text:

Type	Example	Used for
feat_node	"feature node. paper title and abstract: Attention is All You Need..."	Node features
class_node	"prompt node. literature category: cs lo. Logic in CS"	Classification targets
logic_node	"prompt node. either cs lo or cs ai..."	Logic reasoning tasks
Returns 3 task modes:

e2e_node — standard node classification
lr_node — few-shot node classification
logic_e2e — logic-based classification (OR/NOT-AND combinations of classes)
3. Cora/gen_data.py — Cora citation network
Graph: 2,708 paper nodes, 10,556 edges, 7 CS categories.


data = torch.load("cora.pt")        # pre-saved, no download needed
nx_g = pyg.utils.to_networkx(data, to_undirected=True)
edge_index = torch.tensor(list(nx_g.edges())).T
Loads from a local .pt file. Converts via NetworkX to deduplicate edges.


edge_label_text = [
    "prompt node. two papers do not have co-citation",
    "prompt node. two papers have co-citation",
]
logic_label_text = get_logic_label(ordered_desc)
Cora supports the most task types of all citation datasets:

Task	Description
e2e_node	Node classification (7 categories)
e2e_link	Link prediction (co-citation yes/no)
lr_node	Few-shot node classification
logic_e2e	Logic reasoning on category combinations
4. Pubmed/gen_data.py — Pubmed medical papers
Graph: 19,717 paper nodes, 88,648 edges, 3 classes (Diabetes Experimental / Type 1 / Type 2).

Identical structure to Cora but simpler — only 3 classes, no logic task:


with open("categories.csv") as f:
    ordered_desc = f.read().split("\n")   # 3 long Diabetes descriptions
Task modes: e2e_node, lr_node, e2e_link — no logic task (too few classes to combine meaningfully).

5. wikics/gen_data.py — WikiCS
Graph: 11,701 Wikipedia CS article nodes, 216,123 edges, 10 categories.


pyg_data = WikiCS(root=dset.data_dir)   # auto-downloads from PyG
Node text is richer than citation graphs:


node_feature = "feature node. wikipedia entry name: " + node["title"] 
             + ". entry content: " + all_tokens_joined
Uses the full Wikipedia article tokens, not just a title/abstract.


label_feature = "prompt node. wikipedia entry category: " + label
Categories include: Computational linguistics, Databases, Operating systems, Computer architecture, etc.

Task modes: e2e_node, lr_node only — no link task.

6. KG/gen_data.py — FB15K237 + WN18RR
Handles both KG datasets in one file, selected by self.name.


class KGOFADataset(OFAPygDataset):
    def gen_data(self):
        name_dict = {n: path/to/n+".txt" for n in ["train","valid","test"]}
        return read_knowledge_graph(name_dict, self.name)
FB15K237 — Freebase entities:


text = "entity names: " + label + ", alternatives: " + aliases 
     + ". descriptions: " + description   # from entity2wikidata.json
14,541 entities, 237 Freebase relation types (e.g. /people/person/nationality).

WN18RR — WordNet synsets:


text = synset_name + "\t" + definition   # from entity2text.txt
40,943 synsets, 11 lexical relations (hypernym, meronym, similar_to, etc.).

Both datasets produce the same graph structure:


new_data = Data(
    x = zeros(num_entities),         # entity IDs (no numeric features)
    edge_index = train_triplets,      # (head, tail) pairs
    edge_types = relation_ids,        # which relation type each edge is
)
Key difference from citation graphs: edges have types (237 or 11 relation labels), and the task is to predict the relation type, not a node label.

Task modes: e2e_link, lr_link.

7. chemmol/gen_data.py — Molecules
Graph: Each molecule = its own graph. ~500k molecules in ChEMBL pretraining.


graphs, label_text = get_local_text("chemblpre")  # loads from HuggingFace cache
Node features come from SMILES strings via RDKit (gen_raw_graph.py):


atom_feature = [
    element_name,          # "Carbon", "Oxygen"...
    "atomic number is 6",
    "tetrahedral clockwise chirality",
    "degree of 3",
    "formal charge of 0",
    "num of hydrogen is 1",
    "num of radical electrons is 0",
    "hybridization is SP2",
    "is aromatic",
    "is in ring",
]
# → "feature node. atom: Carbon , atomic number is 6 , ..."
Edge features from bonds:


bond_feature = ["AROMATIC bond", "bond stereo is none", "is conjugated"]
# → "feature edge. chemical bond. AROMATIC bond , ..."
Key difference from all other datasets: node/edge texts are deduplicated — many atoms share the same description (e.g. all plain carbons get identical text), so only unique strings are stored and indexed:


node_texts2id = {text: i for i, text in enumerate(unique_node_texts)}
data.x = [node_texts2id[atom_text] for atom in molecule]  # indices, not embeddings
At training time, MolOFADataset.get() looks up the actual embedding: data.node_text_feat = self.node_embs[data.x].

Task modes: e2e_graph, lr_graph — graph-level classification (not node-level).

Summary table
File	Dataset	Graph type	Nodes	Task
single_graph/gen_data.py	router	—	—	routes to below
arxiv/gen_data.py	ogbn-arxiv	1 big directed	169k papers	node cls + logic
Cora/gen_data.py	Cora	1 big undirected	2.7k papers	node cls + link + logic
Pubmed/gen_data.py	Pubmed	1 big undirected	19.7k papers	node cls + link
wikics/gen_data.py	WikiCS	1 big undirected	11.7k pages	node cls
KG/gen_data.py	FB15K237/WN18RR	1 big typed	14.5k/41k entities	link/relation pred
chemmol/gen_data.py	ChEMBL	many small	~10 atoms each	graph cls
