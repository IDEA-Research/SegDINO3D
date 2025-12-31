import json
from typing import Dict
import torch
import torch.nn as nn
from collections import OrderedDict

from segdino3d.utils import is_main_process
from .ema_utils import ModelEma

def clean_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Clean a PyTorch state dictionary by removing the "module." prefix from keys.

    Args:
        state_dict (OrderedDict): The input state dictionary.

    Returns:
        OrderedDict: A new state dictionary with the "module." prefix removed from keys.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v

    return new_state_dict

def resume(model_without_ddp: nn.Module,
           logger: callable,
           cfg: Dict,
           optimizer: callable = None,
           lr_scheduler: callable = None,
           ema_model: nn.Module = None) -> None:
    """Resume from checkpoints. Load model, ema_model, optimizer, scheduler
    
    Args:
        model (nn.Module): Model
        logger (logger): To log information
        args (Dict): Argpaser object
        cfg (Dict): Config dict
        optimizer (optmizer, Optional): If given optimizer, we will load it
        lr_scheduler (lr_scheduler, Optional): If given, will load.
        ema_model (nn.Module): EMA Model
    """
    checkpoint = torch.load(cfg.resume, map_location='cpu')

    loadinfo = model_without_ddp.load_state_dict(
        clean_state_dict(checkpoint['model']), strict=False)
    
    if is_main_process():
        logger.info(f"resume from: {cfg.resume}")
        logger.info(f"loadinfo: {str(loadinfo)}")
    if cfg.use_ema:
        print("load ema model ===>")
        if 'ema_model' in checkpoint:
            if is_main_process():
                logger.info("'ema_model' in checkpoint ")
            ckpt = clean_state_dict(checkpoint['ema_model'])
            for key in ema_model.shadow:
                if key in ckpt:
                    ema_model.shadow[key].copy_(ckpt[key])
        else:
            if is_main_process():
                logger.info("no ema_model in checkpoint ")
        print("end ema loading" + "=" * 10)

    if 'optimizer' in checkpoint and optimizer is not None:
        if len(optimizer.param_groups) == len(
                checkpoint['optimizer']['param_groups']):
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            if is_main_process():
                logger.info("not loading optmizaer",
                            len(optimizer.param_groups))

    if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if 'epoch' in checkpoint:
        cfg.start_epoch = checkpoint['epoch'] + 1

    if 'step' in checkpoint:
        cfg.resume_iter = checkpoint['step'] + 1


def load_model(model_without_ddp: nn.Module,
               logger: callable,
               cfg: Dict,
               ema_model: nn.Module = None) -> None:

    """Load checkpoint for model and ema_model
    """
    checkpoint = torch.load(cfg.load_pretrained_ckpt, map_location='cpu')

    loadinfo = model_without_ddp.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    if is_main_process():
        logger.info(f"loadinfo: {str(loadinfo)}")

    if cfg.use_ema:
        if is_main_process():
            logger.info("load ema model ===>")
        if 'ema_model' in checkpoint:
            if is_main_process():
                logger.info("'ema_model' in checkpoint ")
            ckpt = clean_state_dict(checkpoint['ema_model'])
            for key in ema_model.shadow:
                if key in ckpt:
                    ema_model.shadow[key].copy_(ckpt[key])
        else:
            if is_main_process():
                logger.info("no ema_model in checkpoint ")
            # del ema_model
            # ema_model = ModelEma(model, cfg.ema_decay)

