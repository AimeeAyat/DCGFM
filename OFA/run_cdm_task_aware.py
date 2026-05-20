"""
run_cdm_task_aware.py — Task-type-aware hybrid model runner.

What's different from run_cdm.py (original untouched):
  1. BinGraphAttModel → TaskAwareHybridModel
  2. FewShotDataset.__getitem__ and FewShotDataset_fixed_hard.__getitem__ patched
     to set  g.is_mol = tensor([True])  for mol graphs and  tensor([False])  for
     node/link graphs — used by TaskAwareHybridModel to route each molecule to the
     MoE path and each node/KG graph to the 5L-RGCN baseline path.
  3. training_step patched to add MoE auxiliary load-balancing loss.

Cache safety
============
  Hard pruning indices cache (dcgfm_hard_prune_api_25_0.7_{0,1,2}.pkl) already
  exists from the April 3-task baseline run — all three tasks, near-center.
  No recomputation needed.

  Episode indices (class_ind_*.pkl etc.) are also pre-cached — reused as-is.

  all_no_prompt_data is constructed without is_mol (used for SVDD only, never
  passed to the model) — unaffected by the patch.

Run command:
  python run_cdm_task_aware.py \\
      --override yamls/soft_and_hard_task_aware.yaml \\
      --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 \\
      --prune_ratio 0.3 --hard_pruning_epochs 25 \\
      --hard_pruning_reverse --save_model \\
      --exp_label "all tasks task-aware hybrid near-center"
"""

import os
import json
import argparse
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn

# ── 1. Patch graph constructors to tag is_mol BEFORE any imports use them ────
import ofa_datasets_combine as _odc

def _make_is_mol_patcher(orig_fn, is_mol_value: bool):
    """Wraps a __getitem__ to stamp is_mol onto the returned subgraph."""
    val = torch.tensor([is_mol_value])
    def _patched(self, idx):
        subg = orig_fn(self, idx)
        subg.is_mol = val.clone()
        return subg
    return _patched

# FewShotDataset_fixed_hard (training) and FewShotDataset (eval/test)
# Both store self.query_graph_dataset; GraphListHierDataset = mol data.
for _cls in [_odc.FewShotDataset, _odc.FewShotDataset_fixed_hard]:
    _orig = _cls.__getitem__
    _is_mol_cls = _odc.GraphListHierDataset

    def _make_patched(orig, mol_cls):
        def _patched(self, idx):
            subg = orig(self, idx)
            subg.is_mol = torch.tensor(
                [isinstance(self.query_graph_dataset, mol_cls)]
            )
            return subg
        return _patched

    _cls.__getitem__ = _make_patched(_orig, _is_mol_cls)

print("is_mol patch applied to FewShotDataset and FewShotDataset_fixed_hard")
# ─────────────────────────────────────────────────────────────────────────────

# ── 2. Patch model ────────────────────────────────────────────────────────────
import run_cdm
from models.model_task_aware import TaskAwareHybridModel, AUX_WEIGHT

run_cdm.BinGraphAttModel = TaskAwareHybridModel
run_cdm.BinGraphModel    = TaskAwareHybridModel
run_cdm.PyGRGCNEdge      = lambda *a, **kw: nn.Module()   # placeholder

# ── 3. Patch training_step to add MoE aux loss ────────────────────────────────
from gp.lightning.module_template import IBBaseTemplate

_orig_training_step = IBBaseTemplate.training_step

def _task_aware_training_step(self, batch, batch_idx, dataloader_idx=0):
    ib_loss = _orig_training_step(self, batch, batch_idx, dataloader_idx)
    if (ib_loss is not None
            and hasattr(self.model, 'last_aux_loss')
            and self.model.last_aux_loss is not None):
        ib_loss = ib_loss + AUX_WEIGHT * self.model.last_aux_loss
        self.model.last_aux_loss = None
    return ib_loss

IBBaseTemplate.training_step = _task_aware_training_step
# ─────────────────────────────────────────────────────────────────────────────

# ── 4. Save results.json ──────────────────────────────────────────────────────
from gp.utils.utils import load_yaml, combine_dict, merge_mod, setup_exp, set_random_seed
import gp.lightning.training as _training

_true_orig_summary = _training.dict_res_summary   # capture before any other patches
_exp_dir = [None]

def _summary_and_save(test_col):
    res = _true_orig_summary(test_col)
    results = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in res.items()}
    with open(os.path.join(_exp_dir[0], "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {_exp_dir[0]}/results.json")
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
    run_cdm.main(params)
