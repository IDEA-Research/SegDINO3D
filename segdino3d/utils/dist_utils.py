import io
import os
import functools
import pickle
import torch
import torch.distributed as dist
import json


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """

    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")

    return dist.group.WORLD


def is_dist_avail_and_initialized() -> bool:
    """Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed training is available and initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank() -> int:
    """Get the rank of the current process in the distributed group.

    Returns:
        int: The rank of the current process in the distributed group, or 0 if distributed
        training is not available or initialized.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if the current process is the main process in the distributed group.

    Returns:
        bool: True if the current process is the main process in the distributed group,
            False otherwise.
    """
    return get_rank() == 0


def get_world_size() -> int:
    """Get the number of processes in the distributed group.

    Returns:
        int: The number of processes in the distributed group, or 1 if distributed
            training is not available or initialized.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized() -> bool:
    """Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed training is available and initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def all_gather_cpu(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()],
                              device=device,
                              dtype=torch.long)
    size_list = [
        torch.tensor([0], device=device, dtype=torch.long)
        for _ in range(world_size)
    ]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size, ), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size, ), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy())
        obj = torch.load(buffer)
        data_list.append(obj)

    return data_list


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    if os.getenv("CPU_REDUCE") == "1":
        return all_gather_cpu(data)

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size, ), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size, ), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '': # 'RANK' in os.environ and 
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])

        if os.environ.get('HAND_DEFINE_DIST_URL', 0) == '1':
            pass
        else:
            nodenames = parse_nodelist(os.environ['SLURM_JOB_NODELIST'])
            gpu_ids = [int(node[3:]) for node in nodenames]
            fixid = int(os.environ.get('FIX_DISTRIBUTED_PORT_NUMBER', 0))
            # fixid += random.randint(0, 300)
            port = str(3137 + int(min(gpu_ids)) + fixid)
            args.dist_url = "tcp://{ip}:{port}".format(ip=nodename_to_ip(nodenames[0]), port=port)

        print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank, args.local_rank, torch.cuda.device_count()))


    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print("world_size:{} rank:{} local_rank:{}".format(args.world_size, args.rank, args.local_rank))
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)

    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        world_size=args.world_size, 
        rank=args.rank,
        init_method=args.dist_url,
    )

    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def nodename_to_ip(nodename: str):
    """Convert a node name to its corresponding IP address.

    Args:
        nodename (str): The name of the node to convert to an IP address.

    Returns:
        str: The IP address corresponding to the given node name.

    Raises:
        AssertionError: If the nodename is invalid.
    """
    """
    e.g. 
    Input:
        nodename: dgx020 / 020
    return: 192.168.190.20
    """
    assert len(nodename) == 3 or len(nodename) == 6 , \
        "invalid nodename: {}".format(nodename)
    if len(nodename) == 6:
        nodename = nodename[3:]
    return "192.168.190.{}".format(str(int(nodename)))


def parse_nodelist(nodelist: str):
    """
    e.g.
    Input:
        nodelist(SLURM_NODELIST): 
            e.g:    "dgx[001-002]", 
                    "dgx[074,076-078]"
    return 
        ['dgx001', 'dgx002'], 
        ['dgx074', 'dgx076', 'dgx077', 'dgx078']
    """
    if '[' not in nodelist:
        return [nodelist]
    nodelist = nodelist.split('[')[1].split(']')[0].strip()

    reslist = []
    blocklist = [i.strip() for i in nodelist.split(',')]
    for block in blocklist:
        if '-' in block:
            start_cnt, end_cnt = block.split('-')
            block_nodes = [
                "dgx%03d" % i for i in range(int(start_cnt),
                                             int(end_cnt) + 1)
            ]
        else:
            block_nodes = ["dgx%03d" % int(block)]
        reslist.extend(block_nodes)
    return reslist

