import os
from typing import Dict, Iterable, List

import torch

from utils import (MetricLogger, SmoothedValue, find_unused_parameters,
                   reduce_dict, find_unused_parameters)
from segdino3d.utils import is_main_process

from evaluation import evaluate_3d

def train_multi_loader_step(
    model: torch.nn.Module,
    main_loader: Iterable,
    eval_loaders: List[Iterable],
    evaluator,
    optimizer: torch.optim.Optimizer,
    max_norm: float = 0,
    start_iter: int = 0,
    lr_scheduler=None,
    logger=None,
    args: Dict = None,
    ema_model=None,
    device="cuda",
    log_loss_keys: List[str] = [
        "loss",
        "loss_ce",
        "loss_ce_dn",
        "loss_giou",
        "loss_giou_dn",
        "loss_mask",
        "loss_dice",
    ],
    eval_first=False,
    eval_only=False,
) -> Dict:
    """Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        main_loader (Iterable): The main data loader to use for training.
        extra_loaders (Dict): Extra data loader for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        max_norm (float): The maximum norm for gradient clipping.
        start_iter (int): The start iteration number.
        lr_scheduler: The learning rate scheduler to use.
        logger: The logger to use for logging.
        args: Additional arguments.
        ema_model: The exponential moving average to use for model weights.
        log_loss_leys (List[str]): The list of loss keys to log. If None, all loss keys will be logged.

    Returns:
        Dict: The training results.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr",
                            SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone",
                            SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder",
                            SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr",
                            SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Iteration: "
    print_freq = args.print_freq

    if eval_first or eval_only:
        if args.use_ema:
            ema_model.gather()
            ema_model.apply_shadow()
        evaluator.reset()
        for loader in eval_loaders:
            evaluate_3d(evaluator, loader, args, -1)
        if eval_only:
            return
        if args.use_ema:
            ema_model.restore()
        torch.distributed.barrier()

    model.train()

    curr_step = start_iter

    dtype = torch.bfloat16 if args.amp else torch.float32

    for samples, targets in metric_logger.log_every(
        main_loader,
        print_freq,
        start_iter=start_iter,
        num_iterations=args.num_iterations,
        header=header,
        logger=logger,
    ):
        samples = [s.to(device) for s in samples]
        targets = [t.to(device) for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=dtype):
            loss_dict = model(samples, targets=targets)
            losses = sum(loss_dict.values())

        if args.amp:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
        lr_scheduler.step()

        if args.use_ema:
            ema_model.update()

        main_loss_dict_reduced = reduce_dict(loss_dict)
        main_loss_value = losses.item()
        del loss_dict
        del losses

        if is_main_process():
            log_dict = dict()
            # add main total loss
            log_dict.update(dict(total_loss=main_loss_value))
            
            for key in log_loss_keys:
                if key in main_loss_dict_reduced:
                    log_dict.update({key: main_loss_dict_reduced[key]})
                    
            metric_logger.update(**log_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
            metric_logger.update(
                lr_text_encoder=optimizer.param_groups[2]["lr"])

        if (curr_step + 1) % args.save_step == 0 and args.use_ema:
            ema_model.gather()

        if (curr_step + 1) % args.save_step == 0 and is_main_process():
            weights = {
                "model": model.state_dict(),
                "args": args,
                "optimizer": optimizer.state_dict(),
                "step": curr_step,
            }
            if args.use_ema:
                weights["ema_model"] = ema_model.get_shadow()
            torch.save(
                weights,
                os.path.join(args.work_dir, "checkpoints",
                             f"checkpoint_s{curr_step:010}.pth"),
            )
            print("Saved checkpoint to {}".format(
                os.path.join(
                    args.work_dir,
                    "checkpoints",
                    f"checkpoint_s{curr_step:010}.pth",
                )))

        if (curr_step + 1) % args.eval_step == 0:
            if args.use_ema:
                ema_model.apply_shadow()
            evaluator.reset()
            for loader in eval_loaders:
                evaluate_3d(evaluator, loader, args, curr_step)
            torch.distributed.barrier()
            model.train()
        
        if args.use_ema and ema_model.is_gathered:
            ema_model.restore()

        if (curr_step + 1) == args.num_iterations:
            break
        curr_step += 1

        is_first = False

    return
