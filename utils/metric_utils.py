import datetime
import time
from collections import defaultdict, deque
import torch
import torch.distributed as dist

from segdino3d.utils import is_dist_avail_and_initialized, is_main_process

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64,
                         device='cuda')
        # dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        eps = 1e-6
        return self.total / (self.count + eps)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """A class for logging and tracking metrics during training.

    Attributes:
        meters (defaultdict): A dictionary of SmoothedValue objects for tracking metrics.
        delimiter (str): The delimiter to use when printing the metrics.
    """

    def __init__(self, delimiter="\t"):
        """Initializes a new instance of the MetricLogger class.

        Args:
            delimiter (str, optional): The delimiter to use when printing the metrics. Defaults to "\t".
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Updates the metric values with the given keyword arguments.

        Args:
            **kwargs: The keyword arguments containing the metric values to update.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Gets the value of the specified attribute.

        Args:
            attr (str): The name of the attribute to get.

        Raises:
            AttributeError: If the specified attribute is not found.

        Returns:
            The value of the specified attribute.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """Returns a string representation of the metric values.

        Returns:
            str: A string representation of the metric values.
        """
        loss_str = []
        for name, meter in self.meters.items():
            if meter.count > 0:
                loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        Synchronizes the metric values between processes.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Adds a new meter to the metric logger.

        Args:
            name (str): The name of the meter.
            meter (SmoothedValue): The SmoothedValue object to add.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, start_iter=0, num_iterations=-1, header=None, logger=None):
        """Logs the metric values every `print_freq` iterations.

        Args:
            iterable: The iterable to log.
            print_freq (int): The frequency at which to log the metric values.
            num_iterations (int): The total number of iterations. If -1, then adopt len(iterable).
            header (str, optional): The header to print before the metric values. Defaults to None.
            logger (logging.Logger, optional): The logger to use for printing. Defaults to None.

        Yields:
            The next item in the iterable.
        """
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = start_iter
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}', 'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        if num_iterations < 0:
            num_iterations = len(iterable)
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            # import ipdb; ipdb.set_trace()
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == num_iterations - 1:
                eta_seconds = iter_time.global_avg * (num_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_func(
                        log_msg.format(
                            i,
                            num_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print_func(
                        log_msg.format(
                            i,
                            num_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if is_main_process():
            print_func('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))

