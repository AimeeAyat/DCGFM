**Big Picture**

`G:\Rabia-Salman\DCGFM\OFA\data\chemmol` is not the final dataset itself. It is the code that explains how molecule datasets are built.

In simple words:

- Raw molecule records are loaded from the Hugging Face dataset cache through [`gen_data.py`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_data.py:31>).
- Each molecule is read as a `SMILES` string like a compact text formula for a chemical structure.
- That SMILES is converted into a graph in [`gen_raw_graph.py`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_raw_graph.py:149>).
- In that graph:
  - nodes = atoms
  - edges = bonds
- Each node and edge is turned into a readable sentence, not just a number.
- Then SBERT-like sentence encoding converts those sentences into vectors before the GNN uses them.

So the flow is:

`SMILES -> molecule graph -> atom/bond text -> SBERT embeddings -> GNN`

---

**1. Where the molecules actually come from**

The data loading happens in [`get_local_text()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_data.py:31>).

What it does:

- It loads `haitengzhao/molecule_property_instruction` from local Hugging Face cache.
- It groups rows by `molecule_index`, because one molecule can have multiple assay/property labels.
- For each unique molecule, it takes:
  - the SMILES string
  - the split: train/valid/test
  - the task labels for that molecule

Important point:

- `OFA\data\chemmol` is more like a “dataset builder recipe”.
- The actual cached source dataset is under [`OFA\cache_data\dataset`](</g:/Rabia-Salman/DCGFM/OFA/cache_data/dataset>).
- The processed graph datasets are stored under folders like [`OFA\cache_data\chemhiv`](</g:/Rabia-Salman/DCGFM/OFA/cache_data/chemhiv>) or [`OFA\cache_data\chempcba`](</g:/Rabia-Salman/DCGFM/OFA/cache_data/chempcba>), depending on task.

---

**2. How one molecule becomes a graph**

This happens in [`smiles2graph()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_raw_graph.py:149>).

Layman version:

- A molecule written as SMILES is read by RDKit.
- RDKit understands which atoms exist and which atoms are connected.
- Every atom becomes one node.
- Every bond becomes one edge.
- Because graph neural nets usually pass messages both ways, each bond is stored twice:
  - `atom i -> atom j`
  - `atom j -> atom i`

So if carbon is bonded to oxygen, the code stores both directions, even though chemically it is one bond.

---

**3. How nodes are constructed**

Node text is built in [`atom_to_feature()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_raw_graph.py:94>).

For each atom, the code creates a sentence like:

- element name
- atomic number
- chirality
- degree
- formal charge
- number of hydrogens
- number of radical electrons
- hybridization
- aromatic or not
- ring membership or not

The final node text looks like this pattern:

- `"feature node. atom: carbon, atomic number is 6, tetrahedral clockwise chirality, degree of 4, ..."`

So the node is not stored as just “C”.
It becomes a descriptive sentence.

Why this is done:

- A sentence encoder can understand text better than raw chemistry IDs.
- Similar atoms with similar descriptions get similar embeddings.

Also note:

- Element names come from [`id2element.csv`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/id2element.csv>).
- The atom feature vocabulary is defined near the top of [`gen_raw_graph.py`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_raw_graph.py:17>).

---

**4. How edges are constructed**

Edge text is built in [`bond_to_feature()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_raw_graph.py:118>).

For each bond, the sentence contains:

- bond type: single/double/triple/aromatic
- stereo information: none, E, Z, cis, trans, etc.
- conjugation: conjugated or not

The final edge text looks like:

- `"feature edge. chemical bond. SINGLE bond, bond stereo is none, not conjugated"`

Again, this is not a numeric chemistry feature table only. It is turned into text on purpose.

---

**5. Are subgraphs selected in `chemmol`?**

For molecule classification, usually no subgraph is selected from inside one molecule.

This is the most important confusion to clear up.

There are two different graph-building styles in this repo:

- `SubgraphDataset` logic:
  - samples k-hop neighborhoods around a node
  - used mostly for node/link tasks
  - code is in [`ofa_datasets_combine.py`](</g:/Rabia-Salman/DCGFM/OFA/ofa_datasets_combine.py:230>)
- `GraphListDataset` logic:
  - uses the full graph as one sample
  - used for molecule graph tasks
  - code is in [`GraphListDataset`](</g:/Rabia-Salman/DCGFM/OFA/ofa_datasets_combine.py:459>)

For molecules, `ConstructMolCls()` uses [`GraphListHierDataset`](</g:/Rabia-Salman/DCGFM/OFA/task_constructor.py:273>), not the k-hop subgraph sampler.

That means:

- one molecule = one whole graph
- all atoms in that molecule are included
- all bonds in that molecule are included

So for `chemmol`, “subgraph selection” usually does **not** mean:
“pick 2 hops around one atom”

Instead it means:
“take the whole molecule graph, then attach prompt/query/class nodes around it for the learning task”

---

**6. Then what extra nodes are added?**

This happens in [`make_prompted_graph()`](</g:/Rabia-Salman/DCGFM/OFA/ofa_datasets_combine.py:148>) and the molecule-specific graph wrapper [`GraphListHierDataset`](</g:/Rabia-Salman/DCGFM/OFA/ofa_datasets_combine.py:502>).

The model does not only use atom nodes.
It also adds special prompt nodes:

- one NOI node
  - NOI = node of interest, basically the task anchor node
- class nodes
  - these represent possible labels, like positive/negative assay descriptions

For molecule classification, the final graph becomes something like:

- original atom nodes
- one prompt/NOI node
- one or more class nodes

Then special prompt edges are added:

- atom nodes <-> NOI node
- NOI node <-> class nodes

This is how the task is turned into a graph reasoning problem.

---

**7. What text do these prompt nodes and prompt edges get?**

These texts are built in [`gen_graph()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_data.py:74>).

There are three text groups beyond atom/bond text:

1. Label/class node texts  
Built from assay descriptions using [`get_label_texts()`](</g:/Rabia-Salman/DCGFM/OFA/utils.py:191>)

These look like:

- `"prompt node. molecule property description. The molecule is effective to the following assay. ..."`
- and also negative versions:
- `"The molecule is not effective to the following assay. ..."`

The assay descriptions come from:

- [`mol_label_desc.json`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/mol_label_desc.json>)
- or [`prompt_pretrain.json`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/prompt_pretrain.json>)

2. NOI node texts  
Defined in [`gen_graph()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_data.py:100>)

Examples:

- `"prompt node. graph classification on molecule property"`
- `"prompt node. few shot task node for graph classification that decides whether the query molecule belongs to the class of support molecules."`

3. Prompt edge texts  
Defined in [`gen_graph()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_data.py:98>)

Examples:

- `"prompt edge."`
- `"prompt edge. edge for query graph that is our target"`
- `"prompt edge. edge for support graph that is an example"`

So, not only atoms and bonds have text.
The task machinery also has text.

---

**8. How labels are formed**

Inside [`get_local_text()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_data.py:31>), for each molecule:

- all tasks linked to that molecule are collected
- all Yes/No labels are collected
- a label vector is created
- unknown tasks are kept as `NaN`

So if a dataset has many assays:

- a molecule may be known positive for some assays
- known negative for some assays
- missing for many others

That is why the label vector can contain:

- `1`
- `0`
- `NaN`

This is common in molecular property datasets.

---

**9. Why are unique node/edge texts stored only once?**

This is done in [`gen_graph()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_data.py:74>) and used later in [`MolOFADataset.get()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_data.py:145>).

Reason in plain language:

- Many molecules contain the same kinds of atoms and bonds.
- Example:
  - lots of carbons will have identical text descriptions
  - lots of single bonds will have identical edge text
- Instead of storing the same long sentence again and again for every atom in every molecule, the code:
  - makes one unique text list
  - stores only an index per node/edge
  - later retrieves the real embedding from that index

This saves a lot of memory.

So:

- `data.x` = node text IDs, not final embeddings
- `data.xe` = edge text IDs, not final embeddings
- during `get()`, those IDs are replaced with the corresponding embeddings

---

**10. What exactly is the role of SBERT here?**

This is the core idea.

The sentence encoder is defined in [`SentenceEncoder`](</g:/Rabia-Salman/DCGFM/OFA/utils.py:13>) and called by [`OFAPygDataset.process()`](</g:/Rabia-Salman/DCGFM/OFA/data/ofa_data.py:128>).

Default config uses:

- [`llm_name: "ST"`](</g:/Rabia-Salman/DCGFM/OFA/configs/default_config.yaml:21>)

And `ST` maps to:

- sentence-transformer `multi-qa-distilbert-cos-v1` in [`model.py`](</g:/Rabia-Salman/DCGFM/OFA/models/model.py:293>)

Layman explanation of SBERT’s role:

- The code writes every atom, bond, class label, and prompt edge as English text.
- SBERT reads that text and converts it into a fixed-length vector of numbers.
- That vector is a semantic summary of the text.

So SBERT is doing this kind of conversion:

- `"feature node. atom: oxygen, atomic number is 8, degree of 2, is aromatic ..."`
- becomes
- `[0.13, -0.42, 0.08, ...]` of length 768

Why do this?

Because then the graph model can start from text-aware features instead of hand-designed one-hot chemistry tables only.

In simple terms:

- RDKit tells the system what the atom/bond properties are.
- The code turns those properties into readable sentences.
- SBERT turns those sentences into vectors.
- The GNN learns over the graph using those vectors.

So SBERT is not selecting subgraphs.
SBERT is not deciding which atoms exist.
SBERT is not making bonds.

SBERT’s job is only:

- encode text descriptions into vector features

---

**11. Where the whole-molecule graph is used later**

For molecule tasks, the task constructor calls [`ConstructMolCls()`](</g:/Rabia-Salman/DCGFM/OFA/task_constructor.py:273>), which uses [`GraphListHierDataset`](</g:/Rabia-Salman/DCGFM/OFA/ofa_datasets_combine.py:502>).

That means:

- each dataset item is one full molecule graph
- then prompt nodes and prompt edges are attached
- then the GNN predicts the class/property

For few-shot mode, [`ConstructFSTask()`](</g:/Rabia-Salman/DCGFM/OFA/task_constructor.py:288>) and [`FewShotDataset`](</g:/Rabia-Salman/DCGFM/OFA/ofa_datasets_combine.py:541>) combine:
- one query molecule
- several support molecules
- extra few-shot prompt edges

That is where you see “query graph” and “support graph” wording.

So if you saw subgraph-like language, in molecule few-shot it usually means:
“combine several whole molecule graphs into one larger prompted meta-graph”

not:
“extract a local atom neighborhood from a molecule”

---

**12. Final mental model**

If you want the simplest possible understanding, think of one molecule sample like this:

1. Read molecule SMILES.
2. RDKit breaks it into atoms and bonds.
3. Every atom becomes a node sentence.
4. Every bond becomes an edge sentence.
5. The task description becomes class-node sentences.
6. Prompt helper nodes/edges get their own sentences.
7. SBERT converts all these sentences into vectors.
8. The GNN runs on the final prompted graph.
9. Output says whether the molecule matches the assay/property.

---

**Short answers to your direct questions**

- How are subgraphs selected?  
For `chemmol` molecule classification, usually they are not selected by k-hop sampling. The whole molecule is used. The “subgraph” logic in this repo is mainly for node/link datasets, not standard molecule graph classification.

- How are nodes constructed?  
Each atom becomes a node using RDKit atom information such as element, atomic number, degree, charge, hydrogens, aromaticity, ring status, etc.

- What text do nodes get?  
A descriptive sentence like: atom type + chemistry properties, created by [`atom_to_feature()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_raw_graph.py:94>).

- What text do edges get?  
A descriptive sentence about bond type, stereo, and conjugation, created by [`bond_to_feature()`](</g:/Rabia-Salman/DCGFM/OFA/data/chemmol/gen_raw_graph.py:118>).

- What is SBERT doing?  
Turning those sentences into dense numeric vectors so the graph model can use them as features.

If you want, I can do one more step and walk through a single example molecule end-to-end, showing:
`SMILES -> atom texts -> bond texts -> prompt nodes -> final graph structure`
using one concrete molecule from your local dataset.