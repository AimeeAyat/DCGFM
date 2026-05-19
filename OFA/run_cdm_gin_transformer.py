"""
run_cdm_gin_transformer.py — Design A: 10-layer residual GIN + Transformer.

Patches BinGraphAttModel → GINTransformerModel in run_cdm's namespace.
Original run_cdm.py is untouched.

Run command:
    python run_cdm_gin_transformer.py \\
        --override yamls/soft_and_hard_mol_gin_transformer.yaml \\
        --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 \\
        --prune_ratio 0.3 --hard_pruning_epochs 25
"""

import os
import json
import argparse
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn

import run_cdm
from models.model_gin_transformer import GINTransformerModel

# Patch both model wrappers — GINTransformerModel accepts the same signature as both
run_cdm.BinGraphAttModel = GINTransformerModel
run_cdm.BinGraphModel    = GINTransformerModel
# PyGRGCNEdge is still instantiated by main() but GINTransformerModel ignores it
run_cdm.PyGRGCNEdge = lambda *a, **kw: nn.Module()  # lightweight placeholder

from gp.utils.utils import load_yaml, combine_dict, merge_mod, setup_exp, set_random_seed
import gp.lightning.training as _training

# ── Save results.json at end of training ─────────────────────────────────────
_orig_summary = _training.dict_res_summary
_exp_dir = [None]

def _summary_and_save(test_col):
    res = _orig_summary(test_col)
    results = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in res.items()}
    out_path = os.path.join(_exp_dir[0], "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return res

_training.dict_res_summary = _summary_and_save
# ─────────────────────────────────────────────────────────────────────────────

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
    parser.add_argument("--exp_label",            type=str,   default="", help="Descriptive label appended to the experiment directory name")

    params = parser.parse_args()
    configs = [load_yaml(os.path.join(os.path.dirname(__file__), "configs", "default_config.yaml"))]
    if params.override is not None:
        configs.append(load_yaml(params.override))

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    mod_params.update({k: v for k, v in vars(params).items() if k not in mod_params})

    setup_exp(mod_params)
    if params.exp_label:
        import os as _os
        new_dir = mod_params["exp_dir"] + " " + params.exp_label
        _os.rename(mod_params["exp_dir"], new_dir)
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
    run_cdm.main(params)
