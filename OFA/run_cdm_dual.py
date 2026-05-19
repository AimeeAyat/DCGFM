"""
run_cdm_dual.py — Dual VN-GIN with diameter-split Deep-SVDD pruning.

What's different from run_cdm.py
─────────────────────────────────
1. BinGraphAttModel → DualVNGINModel  (two 10L VN-GINs, hard diameter routing)
2. get_effective_indices → get_effective_indices_dual
       • Computes diameter for every training molecule via scipy BFS
       • Splits molecules at median diameter (≈13) into small / large halves
       • Runs DEEP-SVDD independently on each half → two separate pruning centres
       • Prunes near/far centre within each half (not globally)
       • Merges pruned indices back into one list for training

Why two pruning centres matter
───────────────────────────────
Global SVDD fits one hypersphere to ALL molecules. The centre is a compromise
between compact rings (diam ≤ 5) and long chains (diam > 20). Near-centre
pruning with a global centre discards molecules that are typical for THEIR
structural class but look unusual relative to the global mixture.
Split SVDD gives each regime its own centre, so "near-centre" means "typical
within small molecules" or "typical within large molecules" — not "typical in
the combined soup."

Run command:
    python run_cdm_dual.py \\
        --override yamls/soft_and_hard_mol_dual.yaml \\
        --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 \\
        --prune_ratio 0.3 --hard_pruning_epochs 25 \\
        --hard_pruning_reverse --save_model \\
        --exp_label "mol dual VN 10L split-SVDD near-center"
"""

import os
import json
import pickle
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# ── Patch model ───────────────────────────────────────────────────────────────
import run_cdm
from models.model_dual_gin import DualVNGINModel, DIAMETER_THRESHOLD

run_cdm.BinGraphAttModel = DualVNGINModel
run_cdm.BinGraphModel    = DualVNGINModel
run_cdm.PyGRGCNEdge      = lambda *a, **kw: nn.Module()

# ── Diameter-split pruning ────────────────────────────────────────────────────
from run_cdm import get_useful_indices          # reuse existing SVDD logic
from models.model_adaptive import _compute_batch_diameters
from torch_geometric.data import Batch


def _graph_diameter(data):
    """Diameter of a single PyG Data object via scipy BFS."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path

    n = data.num_nodes
    if n <= 1:
        return 0
    ei = data.edge_index.cpu().numpy()
    adj = sp.csr_matrix(
        (np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)
    )
    dists  = shortest_path(adj, directed=False)
    finite = dists[np.isfinite(dists)]
    return int(finite.max()) if len(finite) > 0 else 0


def get_effective_indices_dual(params, tasks):
    """
    Diameter-split Deep-SVDD hard pruning for a single-task (mol_fs) run.

    Steps:
      1. Compute diameter for every training molecule.
      2. Split into small (≤ THRESHOLD) and large (> THRESHOLD) halves.
      3. Run SVDD independently on each half.
      4. Prune each half by params.hard_pruning_ratio (near or far centre).
      5. Merge and return combined effective indices.
    """
    num_tasks = len(tasks.datasets["train"])

    # Fall back to standard logic for 0-ratio or multi-task
    if params.hard_pruning_ratio == 0.0 or num_tasks != 1:
        return run_cdm.get_effective_indices(params, tasks)

    mode       = params.hard_pruning_mode
    ratio      = params.hard_pruning_ratio
    epochs     = params.hard_pruning_epochs
    reverse    = params.hard_pruning_reverse
    cache_tag  = f"dual_{mode}_{epochs}_{ratio}_{'rev' if reverse else 'fwd'}"

    indices_dir = os.path.join(params.big_data_cache_dir, "hard_pruning_indices")
    os.makedirs(indices_dir, exist_ok=True)
    cache_path  = os.path.join(indices_dir, f"{cache_tag}_0.pkl")

    if os.path.exists(cache_path):
        print(f"Loading dual cached indices from {cache_path}")
        with open(cache_path, "rb") as f:
            effective = pickle.load(f)
        return [effective]

    # ── 1. All training molecules ─────────────────────────────────────────────
    all_data = tasks.datasets["train"][0].all_no_prompt_data
    print(f"Computing diameters for {len(all_data):,} training molecules ...")

    diameters = []
    for i, d in enumerate(all_data):
        if i % 10_000 == 0:
            print(f"  {i:>7,} / {len(all_data):,}", flush=True)
        diameters.append(_graph_diameter(d))
    diameters = np.array(diameters)

    # ── 2. Exact 50/50 split by ordering on diameter ─────────────────────────
    n = len(diameters)
    sorted_by_diam = np.argsort(diameters, kind="stable")   # ascending; stable preserves ties
    small_idx = sorted_by_diam[: n // 2]                    # first  half — smaller diameters
    large_idx = sorted_by_diam[n // 2 :]                    # second half — larger diameters

    # Record the actual split boundary so the model can use it for routing
    split_threshold = int(diameters[sorted_by_diam[n // 2]])
    print(f"50/50 split: {len(small_idx):,} small | {len(large_idx):,} large "
          f"| boundary diameter = {split_threshold}")

    # Save threshold alongside indices so the model loads the same value
    thresh_path = os.path.join(indices_dir, f"{cache_tag}_threshold.pkl")
    with open(thresh_path, "wb") as f:
        pickle.dump(split_threshold, f)

    # ── 3. Independent SVDD pruning on each half ──────────────────────────────
    def _prune_subset(global_indices):
        subset = [all_data[i] for i in global_indices]
        local_useful = get_useful_indices(
            subset, mode, ratio, reverse, epochs
        )
        return global_indices[np.array(local_useful)]

    print("Running SVDD on small-diameter half ...")
    small_kept = _prune_subset(small_idx)
    print(f"  kept {len(small_kept):,} / {len(small_idx):,}")

    print("Running SVDD on large-diameter half ...")
    large_kept = _prune_subset(large_idx)
    print(f"  kept {len(large_kept):,} / {len(large_idx):,}")

    # ── 4. Merge ──────────────────────────────────────────────────────────────
    effective = np.sort(np.concatenate([small_kept, large_kept])).tolist()
    print(f"Total kept: {len(effective):,} molecules")

    with open(cache_path, "wb") as f:
        pickle.dump(effective, f)
    print(f"Dual pruning indices cached to {cache_path}")

    return [effective]


# ── Patch get_effective_indices in run_cdm ────────────────────────────────────
run_cdm.get_effective_indices = get_effective_indices_dual

# ── Results saving ────────────────────────────────────────────────────────────
from gp.utils.utils import load_yaml, combine_dict, merge_mod, setup_exp, set_random_seed
import gp.lightning.training as _training

_orig_summary = _training.dict_res_summary
_exp_dir = [None]

def _summary_and_save(test_col):
    res = _orig_summary(test_col)
    results = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in res.items()}
    with open(os.path.join(_exp_dir[0], "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {_exp_dir[0]}/results.json")
    return res

_training.dict_res_summary = _summary_and_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)

    parser.add_argument("--control_gpu",          action="store_true")
    parser.add_argument("--gpus",                 type=str,   default="0,1")
    parser.add_argument("--fs_sample_size",       default=60000, type=int)
    parser.add_argument("--save_model",           action="store_true")
    parser.add_argument("--big_data_cache_dir",   type=str,   default="./cache_data/ofa_dataset")
    parser.add_argument("--checkpoint_interval",  default=10, type=int)
    parser.add_argument("--use_onecycle",         action="store_true")
    parser.add_argument("--max-lr",               default=0.05,   type=float)
    parser.add_argument("--div-factor",           default=25,     type=float)
    parser.add_argument("--final-div",            default=10000,  type=float)
    parser.add_argument("--pct-start",            default=0.3,    type=float)
    parser.add_argument("--prune_ratio",          default=0.5,    type=float)
    parser.add_argument("--delta",                default=1.0,    type=float)
    parser.add_argument("--hard_pruning_mode",    type=str,   default="random")
    parser.add_argument("--hard_pruning_ratio",   type=float, default=0)
    parser.add_argument("--hard_pruning_epochs",  type=int,   default=25)
    parser.add_argument("--hard_pruning_joint",   action="store_true")
    parser.add_argument("--hard_pruning_reverse", action="store_true")
    parser.add_argument("--ckpt_path",            type=str,   default=None)
    parser.add_argument("--task_config",          type=str,   default="configs/task_config.yaml")
    parser.add_argument("--exp_label",            type=str,   default="")

    params = parser.parse_args()
    configs = [load_yaml(os.path.join(os.path.dirname(__file__), "configs", "default_config.yaml"))]
    if params.override is not None:
        configs.append(load_yaml(params.override))

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    mod_params.update({k: v for k, v in vars(params).items() if k not in mod_params})

    setup_exp(mod_params)
    if params.exp_label:
        new_dir = mod_params["exp_dir"] + " " + params.exp_label
        os.rename(mod_params["exp_dir"], new_dir)
        mod_params["exp_dir"] = new_dir
    _exp_dir[0] = mod_params["exp_dir"]

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.log_project = "OFA_dcgfm"
    params.exp_name += (
        f"dcgfm_{params.hard_pruning_mode}_{params.hard_pruning_epochs}"
        f"_{params.hard_pruning_ratio}_{params.prune_ratio}"
    )

    if params.control_gpu:
        params.gpus = [int(gpu_id) for gpu_id in params.gpus.split(",")]

    print(params)

    # Load the data-driven split threshold and inject into model post-construction.
    # run_cdm.main() builds the model internally; we patch BinGraphAttModel so that
    # every DualVNGINModel instance gets the correct threshold after __init__.
    _thresh_path = os.path.join(
        params.big_data_cache_dir, "hard_pruning_indices",
        f"dual_{params.hard_pruning_mode}_{params.hard_pruning_epochs}"
        f"_{params.hard_pruning_ratio}_{'rev' if params.hard_pruning_reverse else 'fwd'}_threshold.pkl"
    )
    if os.path.exists(_thresh_path):
        with open(_thresh_path, "rb") as _f:
            _threshold = pickle.load(_f)
        _OrigDualInit = DualVNGINModel.__init__
        def _patched_init(self, *a, **kw):
            _OrigDualInit(self, *a, **kw)
            self.set_diameter_threshold(_threshold)
        DualVNGINModel.__init__ = _patched_init

    run_cdm.main(params)
