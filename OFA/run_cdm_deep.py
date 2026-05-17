"""
run_cdm_deep.py — Deep GIN variant of run_cdm.py.

Only difference from run_cdm.py:
  PyGRGCNEdge  →  PyGRGCNEdgeDeep  (residual skip connections)

Use with:
  python run_cdm_deep.py --override yamls/soft_and_hard_mol_deep.yaml \
      --hard_pruning_mode hard_prune_api --hard_pruning_ratio 0.7 \
      --prune_ratio 0.3 --hard_pruning_epochs 25

The original run_cdm.py is untouched.
"""

import os
import argparse
from types import SimpleNamespace
import torch

# ── Patch run_cdm's GNN class BEFORE calling main() ─────────────────────────
# run_cdm.main() looks up PyGRGCNEdge from its own module globals at call time,
# so replacing the name here redirects it to the residual variant.
import run_cdm
from models.model_deep import PyGRGCNEdgeDeep

run_cdm.PyGRGCNEdge = PyGRGCNEdgeDeep
# ─────────────────────────────────────────────────────────────────────────────

from gp.utils.utils import load_yaml, combine_dict, merge_mod, setup_exp, set_random_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    parser.add_argument("--control_gpu", action="store_true")
    parser.add_argument("--gpus", type=str, default="0,1")

    parser.add_argument("--fs_sample_size", default=60000, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--big_data_cache_dir", type=str, default="./cache_data/ofa_dataset")
    parser.add_argument("--checkpoint_interval", default=10, type=int)

    parser.add_argument("--use_onecycle", action="store_true")
    parser.add_argument("--max-lr", default=0.05, type=float)
    parser.add_argument("--div-factor", default=25, type=float)
    parser.add_argument("--final-div", default=10000, type=float)
    parser.add_argument("--pct-start", default=0.3, type=float)

    parser.add_argument("--prune_ratio", default=0.5, type=float)
    parser.add_argument("--delta", default=1.0, type=float)

    parser.add_argument("--hard_pruning_mode", type=str, default="random")
    parser.add_argument("--hard_pruning_ratio", type=float, default=0)
    parser.add_argument("--hard_pruning_epochs", type=int, default=25)
    parser.add_argument("--hard_pruning_joint", action="store_true")
    parser.add_argument("--hard_pruning_reverse", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--task_config", type=str, default="configs/task_config.yaml")

    params = parser.parse_args()
    configs = [load_yaml(os.path.join(os.path.dirname(__file__), "configs", "default_config.yaml"))]

    if params.override is not None:
        configs.append(load_yaml(params.override))

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    mod_params.update({k: v for k, v in vars(params).items() if k not in mod_params})

    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.log_project = "OFA_dcgfm"
    params.exp_name += f"dcgfm_{params.hard_pruning_mode}_{params.hard_pruning_epochs}_{params.hard_pruning_ratio}_{params.prune_ratio}"

    if params.control_gpu:
        params.gpus = [int(gpu_id) for gpu_id in params.gpus.split(",")]

    print(params)
    run_cdm.main(params)   # calls original main() with PyGRGCNEdgeDeep patched in
