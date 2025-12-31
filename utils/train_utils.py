import torch
import torch.distributed as dist

from segdino3d.utils import get_world_size, is_main_process
from .common_utils import get_param_dict

def adjust_learning_rate(
    optimizer,
    epoch: int,
    curr_step: int,
    args,
):
    """Adjust the lr according to the schedule.

    Args:
        Optimizer: torch optimizer to update.
        epoch(int): number of the current epoch.
        curr_step(int): number of optimization step taken so far.
        num_training_step(int): total number of optimization steps.
        args: additional training dependent args:
              - lr_drop(int): number of epochs before dropping the learning rate.
              - fraction_warmup_steps(float) fraction of steps over which the lr will be increased to its peak.
              - lr(float): base learning rate
              - lr_backbone(float): learning rate of the backbone
              - text_encoder_backbone(float): learning rate of the text encoder
              - schedule(str): the requested learning rate schedule:
                   "step": all lrs divided by 10 after lr_drop epochs
                   "multistep": divided by 2 after lr_drop epochs, then by 2 after every 50 epochs
                   "linear_with_warmup": same as "step" for backbone + transformer, but for the text encoder, linearly
                                         increase for a fraction of the training, then linearly decrease back to 0.
                   "all_linear_with_warmup": same as "linear_with_warmup" for all learning rates involved.

    """
    try:
        num_warmup_steps = args.num_warmup_steps
    except:
        return

    if epoch > 0:
        return

    if curr_step > num_warmup_steps:
        return

    text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
    optimizer.param_groups[-1]["lr"] = args.lr_backbone * text_encoder_gamma


def reduce_dict(input_dict, average=True):
    """Reduce the values in the dictionary from all processes so that all processes have
    the averaged results.

    Args:
        input_dict (dict): A dictionary containing the values to be reduced.
        average (bool): Whether to do average or sum. Defaults to True.

    Returns:
        dict: A dictionary with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def find_unused_parameters(model, samples, targets, amp, dtype):
    with torch.cuda.amp.autocast(enabled=amp, dtype=dtype):
        loss_dict = model(samples, targets=targets)
        losses = sum(loss_dict.values())
    losses.backward()

    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params.append([name, param])
    return params


def build_optimizer_scheduler(cfg, model_without_ddp, logger):
    param_dicts, param_name_dicts = get_param_dict(cfg, model_without_ddp)
    #logger.info("param_dicts:\n" + json.dumps(param_name_dicts, indent=2))
    if cfg.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay)
    else:
        raise NotImplementedError(
            f'optimizer {cfg.optimizer.type} is not implement')

    # build scheduler
    if cfg.scheduler.type == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       cfg.scheduler.lr_drop,
                                                       gamma=cfg.scheduler.gamma if 'gamma' in cfg.scheduler else 0.1)
    elif cfg.scheduler.type == "PolyLR":
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,
                                                        cfg.scheduler.total_iters,
                                                        power=cfg.scheduler.power)
    else:
        raise NotImplementedError(
            f'scheduler {cfg.scheduler.type} is not implement')
    return optimizer, lr_scheduler
