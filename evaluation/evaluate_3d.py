import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pickle
import numpy as np
import pycocotools.mask as mask_utils
from segdino3d.utils import is_main_process, get_rank
from utils.dataset_utils import collate_fn_3D
import torch.nn.functional as F

from .evaluator_3d import InstanceSeg3DEvaluator
from segdino3d.datasets.dataset import ScanNet200InstanceSeg3D
from segdino3d.datasets.dataset import ScanNetInstanceSeg3D


def build_evaluate_datasets_3d(cfg):
    loaders = []
    for key in cfg.evaluations:
        if 'scannet200_instance_seg' in key:
            mode = 'scannet200_instance_seg'
        elif 'scannet_instance_seg' in key:
            mode = 'scannet_instance_seg'
        else:
            raise NotImplementedError
        if mode == 'scannet200_instance_seg':
            dataset = ScanNet200InstanceSeg3D(
                **cfg.data.eval_main[0]
            )
        elif mode == 'scannet_instance_seg':
            dataset = ScanNetInstanceSeg3D(
                **cfg.data.eval_main[0]
            )
        loader = DataLoader(dataset, batch_size=1, num_workers=cfg.data.num_workers, shuffle=False, drop_last=False, collate_fn=collate_fn_3D)
        loaders.append(loader)
    return loaders


def build_evaluator_3d(model, cfg):
    return InstanceSeg3DEvaluator(model, **cfg.evaluator_cfg) 


def evaluate_3d(evaluator, loader, cfg, current_iter, device="cuda"):
    print("Not support multi-card evaluation.")
    for sample_id, (samples, targets) in enumerate(tqdm(loader)):
        res = evaluator.inference_single(samples, targets, device=device)
        results = []  # align format with evaluator
        for res_ in res:
            num_instance = res_["masks"].shape[0]
            pts_instance_mask = res_["masks"].squeeze(-1) * (torch.arange(num_instance))[:, None].to(res_["masks"].device)
            pts_instance_mask = pts_instance_mask.sum(dim=0)
            pts_instance_mask[res_["masks"].squeeze(-1).sum(dim=0) == 0] = -1  # bg is set as num_instance
            pts_semantic_mask = res_["masks"].squeeze(-1) * res_["labels"][:, None]  # class ids is set to 0-199,
            pts_semantic_mask = pts_semantic_mask.sum(dim=0)
            pts_semantic_mask[res_["masks"].squeeze(-1).sum(dim=0) == 0] = loader.dataset.bg_class_id  # bg is set as 200
            result = {
                "eval_ann_info":{
                    "pts_instance_mask": pts_instance_mask.cpu().numpy(),
                    "pts_semantic_mask": pts_semantic_mask.cpu().numpy(),
                    "sp_pts_mask": res_["extra_features"]["super_point_masks"].cpu().numpy(),
                    "lidar_idx": res_["scene_id"],
                },
                "pred_pts_seg": res_.pred_pts_seg
            }
            results.append(result)
        evaluator.process(None, results)
    evaluator.evaluate(len(loader))
