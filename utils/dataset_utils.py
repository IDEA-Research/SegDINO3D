import bisect
from typing import Iterator, List, Tuple, Dict, Any
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, BatchSampler
from segdino3d import build_dataset
from segdino3d.models.module import NestedTensor, nested_tensor_from_tensor_list

class CustomConcatDatasetWithSyncScale(ConcatDataset):

    def __getitem__(self, idx_scale):
        if len(idx_scale) == 2:
            # this for sync scale
            idx, current_scale = idx_scale
        else:
            idx = idx_scale
            current_scale = None
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = (sample_idx, current_scale)
        return self.datasets[dataset_idx][sample_idx]


class CustomBatchSamplerWithSyncScale(BatchSampler):
    """A custom batch sampler that yields batches of indices and a random scale.
    This is used for DDP synchronization in data augmentation.

    Attributes:
        sampler (Sampler): The base sampler.
        batch_size (int): The size of the batches to yield.
        drop_last (bool): Whether to drop the last batch if it's smaller than `batch_size`.
    """

    def __init__(self, sampler: torch.utils.data.Sampler, batch_size: int,
                 drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)
        # define random scale generator with seed 0
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(0)

    def __iter__(self) -> Iterator[Tuple[List[int], float]]:
        """
        Yield batches of indices and a random scale.

        Yields:
            Iterator[Tuple[List[int], int]]: An iterator that yields tuples
            where the first element is a list of indices and the second element is a random scale.
        """
        batch = []
        random_scale = torch.rand(
            (1,),
            generator=self.random_generator).item()

        for idx in self.sampler:
            batch.append((idx, random_scale))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class CustomBatchSamplerWithDDPSyncScale(BatchSampler):
    """A custom batch sampler that yields batches of indices and a random scale.
    This is used for DDP synchronization in data augmentation.

    Attributes:
        sampler (Sampler): The base sampler.
        batch_size (int): The size of the batches to yield.
        drop_last (bool): Whether to drop the last batch if it's smaller than `batch_size`.
    """

    def __init__(self, sampler: torch.utils.data.Sampler, batch_size: int,
                 drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

        # define random scale generator with seed 0
        self.rank = dist.get_rank()  # Get the rank of the current process
        self.world_size = dist.get_world_size()  # Total number of processes
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(0)

    def __iter__(self) -> Iterator[Tuple[List[int], float]]:
        """
        Yield batches of indices and a random scale.

        Yields:
            Iterator[Tuple[List[int], int]]: An iterator that yields tuples
            where the first element is a list of indices and the second element is a random scale.
        """
        batch = []
        for idx in self.sampler:
            if len(batch
                   ) == 0:  # Generate a new scale at the start of each batch
                random_scale = torch.rand(
                    (1,),
                    generator=self.random_generator).cuda()

                random_scale = random_scale.item()

            batch.append((idx, random_scale))
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[NestedTensor, Dict[str, Any]]:
    """Collate function for creating a batch of data from a list of samples.

    Args:
        batch (List[Tuple[torch.Tensor, Dict[str, Any]]]): A list of samples, where
        each sample is a tuple containing a tensor and a dictionary of metadata.

    Returns:
        Tuple[NestedTensor, Dict[str, Any]]: A tuple containing a NestedTenso
          object representing the batch of input tensors, and a dictionary of metadata for the batch.
    """
    batch = list(zip(*batch))
    if isinstance(batch[0][0], torch.Tensor):
        batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


class RepeatingLoader:

    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)

    @property
    def batch_sampler(self):
        return self.loader.batch_sampler

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch

def build_training_datasets(cfg, logger, dataset_cfgs=None):
    dataset_list = []
    if dataset_cfgs is None:
        dataset_cfgs = cfg.data.train
    for dataset_cfg in dataset_cfgs:
        dataset = build_dataset(dataset_cfg)
        logger.info(
            f"Build {dataset_cfg.type} dataset for training, total {len(dataset)} images"
        )
        dataset_list.append(dataset)
    sync_scale = cfg.data.sync_scale if hasattr(
        cfg.data, "sync_scale") else None
    if sync_scale is None:
        dataset_train = ConcatDataset(dataset_list)
    else:
        dataset_train = CustomConcatDatasetWithSyncScale(dataset_list)
    logger.info(f"total len(dataset_train):{len(dataset_train)}")

    if cfg.distributed:
        sampler_train = DistributedSampler(dataset_train, seed=cfg.seed)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if sync_scale is None:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.data.train_batch_size, drop_last=True)
    else:
        if cfg.distributed:
            batch_sampler_train = CustomBatchSamplerWithDDPSyncScale(
                sampler_train,
                cfg.data.train_batch_size,
                drop_last=True)
        else:
            batch_sampler_train = CustomBatchSamplerWithSyncScale(
                sampler_train,
                cfg.data.train_batch_size,
                drop_last=True)
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=lambda batch: collate_fn(batch),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return data_loader_train


def build_iterable_training_datasets(cfg, logger, dataset_cfgs=None):
    data_loader_train = build_training_datasets(cfg, logger, dataset_cfgs)
    iterable_data_loader_train = RepeatingLoader(data_loader_train)
    return iterable_data_loader_train


def collate_fn_3D(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[NestedTensor, Dict[str, Any]]:
    """Collate function for creating a batch of data from a list of samples.

    Args:
        batch (List[Tuple[torch.Tensor, Dict[str, Any]]]): A list of samples, where
        each sample is a tuple containing a tensor and a dictionary of metadata.

    Returns:
        Tuple[NestedTensor, Dict[str, Any]]: A tuple containing a NestedTenso
          object representing the batch of input tensors, and a dictionary of metadata for the batch.
    """
    batch = list(zip(*batch))
    # if isinstance(batch[0][0], torch.Tensor):
    #     batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def build_3D_training_datasets(cfg, logger, dataset_cfgs=None):
    dataset_list = []
    if dataset_cfgs is None:
        dataset_cfgs = cfg.data.train
    for dataset_cfg in dataset_cfgs:
        dataset = build_dataset(dataset_cfg)
        logger.info(
            f"Build {dataset_cfg.type} dataset for training, total {len(dataset)} images"
        )
        dataset_list.append(dataset)
    sync_scale = cfg.data.sync_scale if hasattr(
        cfg.data, "sync_scale") else None
    if sync_scale is None:
        dataset_train = ConcatDataset(dataset_list)
    else:
        dataset_train = CustomConcatDatasetWithSyncScale(dataset_list)
    logger.info(f"total len(dataset_train):{len(dataset_train)}")

    if cfg.distributed:
        sampler_train = DistributedSampler(dataset_train, seed=cfg.seed)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if sync_scale is None:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.data.train_batch_size, drop_last=True)
    else:
        if cfg.distributed:
            batch_sampler_train = CustomBatchSamplerWithDDPSyncScale(
                sampler_train,
                cfg.data.train_batch_size,
                drop_last=True)
        else:
            batch_sampler_train = CustomBatchSamplerWithSyncScale(
                sampler_train,
                cfg.data.train_batch_size,
                drop_last=True)
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=lambda batch: collate_fn_3D(batch),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return data_loader_train


def build_iterable_3D_training_datasets(cfg, logger, dataset_cfgs=None):
    data_loader_train = build_3D_training_datasets(cfg, logger, dataset_cfgs)
    iterable_data_loader_train = RepeatingLoader(data_loader_train)
    return iterable_data_loader_train