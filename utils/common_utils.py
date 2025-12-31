import os
import json
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn

from segdino3d.utils import is_main_process

def get_sha():
    """Get the SHA, status, and branch of the current Git repository.

    Returns:
        str: A string containing the SHA, status, and branch of the current Git repository.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(
            command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def set_seed(seed: int):
    """Set torch.manual_seed, np.random, random seed for reproduction

    Args:
        seed (int)L Randon Seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_param_dict(cfg, model_without_ddp: nn.Module):
    try:
        param_dict_type = cfg.optimizer.param_dict_type
    except:
        param_dict_type = 'default'
    assert param_dict_type in ['default', 'weight_decay', 'vit']

    if not hasattr(cfg.optimizer, 'lr_text'):
        cfg.optimizer.lr_text = cfg.optimizer.lr_backbone

    # by default
    if param_dict_type == 'default':
        param_dicts = [{
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "clip" not in n and "bert" not in n and p.requires_grad
            ]
        }, {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr":
            cfg.optimizer.lr_backbone,
        }, {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if ("bert" in n or "clip" in n) and p.requires_grad
            ],
            "lr":
            cfg.optimizer.lr_text,
        }]

        param_name_dicts = [{
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "clip" not in n and "bert" not in n and p.requires_grad
            ]
        }, {
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr":
            cfg.optimizer.lr_backbone,
        }, {
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if ("bert" in n or "clip" in n) and p.requires_grad
            ],
            "lr":
            cfg.optimizer.lr_text,
        }]
        if is_main_process():
            #print('param_name_dicts: ', json.dumps(param_name_dicts, indent=2))
            pass
        return param_dicts, param_name_dicts

    # by weight decay
    elif param_dict_type == 'weight_decay':
        param_dicts1 = [{
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "clip" not in n and "bert" not in n and "norm" not in n and "bias" not in n and p.requires_grad
            ]
        }, {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and "norm" not in n and "bias" not in n and p.requires_grad
            ],
            "lr":
            cfg.optimizer.lr_backbone,
        }, {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if ("bert" in n or "clip" in n) and "norm" not in n and "bias" not in n and p.requires_grad
            ],
            "lr":
            cfg.optimizer.lr_text,
        }]

        param_name_dicts1 = [{
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "clip" not in n and "bert" not in n and "norm" not in n and "bias" not in n and p.requires_grad
            ]
        }, {
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and "norm" not in n and "bias" not in n and p.requires_grad
            ],
            "lr":cfg.optimizer.lr_backbone
        }, {
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if ("bert" in n or "clip" in n) and "norm" not in n and "bias" not in n and p.requires_grad
            ],
            "lr": cfg.optimizer.lr_text
        }]

        param_dicts2 = [{
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "clip" not in n and "bert" not in n and ("norm" in n or "bias" in n) and p.requires_grad
            ],
            "weight_decay": 0.0
        }, {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and ("norm" in n or "bias" in n) and p.requires_grad
            ],
            "lr":cfg.optimizer.lr_backbone,
            "weight_decay": 0.0
        }, {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if ("bert" in n or "clip" in n) and ("norm" in n or "bias" in n) and p.requires_grad
            ],
            "lr":cfg.optimizer.lr_text,
            "weight_decay": 0.0
        }]

        param_name_dicts2 = [{
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "clip" not in n and "bert" not in n and ("norm" in n or "bias" in n) and p.requires_grad
            ],
            "weight_decay": 0.0
        }, {
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and ("norm" in n or "bias" in n) and p.requires_grad
            ],
            "lr":cfg.optimizer.lr_backbone,
            "weight_decay": 0.0
        }, {
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if ("bert" in n or "clip" in n) and ("norm" in n or "bias" in n) and p.requires_grad
            ],
            "lr":cfg.optimizer.lr_text,
            "weight_decay": 0.0
        }]

        param_dicts = param_dicts1 + param_dicts2
        param_name_dicts = param_name_dicts1 + param_name_dicts2
        if is_main_process():
            #print('param_name_dicts: ', json.dumps(param_name_dicts, indent=2))
            pass
        return param_dicts, param_name_dicts

    elif param_dict_type == 'vit':
        depth = model_without_ddp.backbone.depth
        lr_decay_rate = model_without_ddp.backbone.lr_decay_rate
        param_dicts = [{
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "bert" not in n and p.requires_grad
            ]
        },{
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "bert" in n and p.requires_grad
            ],
            "lr":
            cfg.optimizer.lr_text,
        }]
        param_dicts.extend(
            [
                {"params": p, "lr": cfg.optimizer.lr_backbone * get_vit_lr_decay_rate(n, lr_decay_rate, depth)}
                for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad
            ]
        )
        
        param_name_dicts = [{
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "bert" not in n and p.requires_grad
            ]
        }, {
            "params": [
                n for n, p in model_without_ddp.named_parameters()
                if "bert" in n and p.requires_grad
            ],
            "lr":
            cfg.optimizer.lr_text,
        }]
        param_name_dicts.extend(
            [
                {"params": n, "lr": cfg.optimizer.lr_backbone * get_vit_lr_decay_rate(n, lr_decay_rate, depth)}
                for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad
            ]
        )
        if is_main_process():
            #print('param_name_dicts: ', json.dumps(param_name_dicts, indent=2))
            pass
        return param_dicts, param_name_dicts
    else:
        raise NotImplementedError

