"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""
import math
import mindspore
from mindspore import ops, context
import mindspore.dataset as ds
from mindspore.dataset import Sampler
import numpy as np
from mindspore.ops import AllGather as allgather
from mindspore.ops import AllReduce as allreduce
from mindspore.communication import GlobalComm
from mindspore.communication.management import get_group_size, get_rank, init, get_local_rank

__all__ = ['get_world_size', 'get_rank', 'synchronize', 'is_main_process',
           'all_gather', 'make_data_sampler', 'make_batch_data_sampler',
           'reduce_dict', 'reduce_loss_dict']


def get_world_size():
    if GlobalComm.WORLD_COMM_GROUP is not None:
        return get_group_size(GlobalComm.WORLD_COMM_GROUP)
    return 1


def get_rank():
    if GlobalComm.WORLD_COMM_GROUP is not None:
        return get_rank(GlobalComm.WORLD_COMM_GROUP)
    return 0


def get_local_rank():
    if GlobalComm.WORLD_COMM_GROUP is not None:
        return get_local_rank()
    return 0


def is_main_process():
    return get_rank() == 0


def synchronize():
    if GlobalComm.WORLD_COMM_GROUP is not None:
        init()


from mindspore import Tensor


def all_gather(data):
    """
    Run all_gather on tensors
    Args:
        data: Tensor
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # obtain Tensor size of each rank
    local_size = data.shape
    size_list = [Tensor([0]) for _ in range(world_size)]
    allgather(size_list, local_size)
    size_list = [int(size.asnumpy()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = []
    for _ in size_list:
        tensor_list.append(Tensor(max_size))
    if local_size != max_size:
        padding = Tensor(max_size - local_size)
        data = ops.concat((data, padding))
    allgather(tensor_list, data)

    return tensor_list


def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with mindspore.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = ops.stack(values)
        allreduce(values)
        if get_rank() == 0 and average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with mindspore.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = ops.stack(all_losses)
        allreduce(all_losses)
        if get_rank() == 0:
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = ds.RandomSampler(dataset)
    else:
        sampler = ds.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = ds.BatchSampler(sampler, images_per_batch, drop_last=True)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler


# Code is copy-pasted from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/distributed.py
class DistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not context.get_context("enable_ge"):
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = get_group_size()
        if rank is None:
            if not context.get_context("enable_ge"):
                raise RuntimeError("Requires distributed package to be available")
            rank = get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = list(range(len(self.dataset)))
            np.random.shuffle(indices)
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class IterationBasedBatchSampler(Sampler):
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            if hasattr(self.batch_sampler, "set_epoch"):
                self.batch_sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


if __name__ == '__main__':
    pass
