import argparse
import datetime
import json
import os
import os.path as osp
import sys
import time
import hashlib

import mmengine
import torch
from mmengine.config import Config, DictAction
from transformers.utils import logging

logging.set_verbosity(40)

from engine import train_multi_loader_step_3d
from segdino3d import build_architecture
from segdino3d.utils import get_rank, init_distributed_mode, is_main_process
from utils import (
    ModelEma,
    set_seed, setup_logger_and_init_log,
    code_dumper,
    build_iterable_3D_training_datasets,
    build_optimizer_scheduler,
    resume,
    load_model,
)
from evaluation import (
    build_evaluate_datasets_3d,
    build_evaluator_3d,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser(description='Train DINOX')
    parser.add_argument(
        '--config_file',
        type=str,
        default='config.py',
        help='path to prototype config file')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing')
    parser.add_argument(
        '--eval_first',
        action='store_true',
        help='whether to evaluate before training')
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='whether to evaluate before training')
    parser.add_argument(
        '--resume',
        default=None,
        help='path to latest checkpoint (default: none)')
    parser.add_argument(
        '--load_pretrained_ckpt',
        default=None,
        help='Path to pretrained model checkpoint')
    parser.add_argument(
        '--work_dir',
        default=None,
        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int, help='training seed')
    # distributed training parameters
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # crete work_dir
    if not args.work_dir:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        args.work_dir = osp.join('work_dirs',
                                 osp.basename(args.config_file).split('.')[0],
                                 timestamp)

    init_distributed_mode(args)

    # intialize seed
    seed = args.seed + get_rank()
    args.rank = get_rank()
    set_seed(seed)

    # load config and setup logger
    cfg = Config.fromfile(args.config_file)
    for k, v in args.__dict__.items():
        setattr(cfg, k, v)
    # merge custom options if necessary
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if is_main_process():
        mmengine.mkdir_or_exist(osp.abspath(args.work_dir))
        mmengine.mkdir_or_exist(osp.join(args.work_dir, 'inference_results'))
        mmengine.mkdir_or_exist(osp.join(args.work_dir, 'inference_results', 'cache'))
        mmengine.mkdir_or_exist(osp.join(args.work_dir, 'checkpoints'))
        cfg.dump(osp.join(args.work_dir, 'config.py'))

    # setup logger
    logger = setup_logger_and_init_log(cfg)

    # dump code
    if is_main_process():
        logger.info("Dumping code...")
        code_dumper(args)

    # build model
    device = torch.device(cfg.device)
    model = build_architecture(cfg.model)
    model.to(device)

    # ema model
    ema_model = ModelEma(model, cfg.ema_decay, seed=hashlib.md5(open(args.config_file, 'rb').read()).hexdigest()) if cfg.use_ema else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer
    optimizer, lr_scheduler = build_optimizer_scheduler(
        cfg, model_without_ddp, logger)

    # build training datasets
    main_loader = build_iterable_3D_training_datasets(cfg, logger, cfg.data.train_main)

    # build evaluaion
    eval_loaders = build_evaluate_datasets_3d(cfg)
    evaluator = build_evaluator_3d(model_without_ddp, cfg)

    # resume from checkpoints
    if cfg.resume:
        resume(
            model_without_ddp=model_without_ddp,
            logger=logger,
            cfg=cfg,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
        )

    # load from pretrained checkpoint
    if not cfg.resume and cfg.load_pretrained_ckpt:
        load_model(
            model_without_ddp=model_without_ddp, 
            logger=logger,
            cfg=cfg,
            ema_model=ema_model,
        )
    
    # start training
    if is_main_process():
        logger.info('Start training now')

    start_time = time.time()

    train_stats = train_multi_loader_step_3d(
        model=model,
        main_loader=main_loader,
        eval_loaders=eval_loaders,
        evaluator=evaluator, 
        optimizer=optimizer,
        max_norm=cfg.clip_max_norm,
        start_iter = 0 if not cfg.resume else cfg.resume_iter,
        lr_scheduler=lr_scheduler,
        logger=logger,
        args=cfg,
        ema_model=ema_model,
        eval_first=cfg.eval_first,
        eval_only=cfg.eval_only,
        log_loss_keys=[
            "seg_loss",
            "inst_loss",
        ]
    )
    if not cfg.eval_only:
        # finish
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if is_main_process():
            logger.info('Training time {}'.format(total_time_str))
