import torch
import random


class RandomCycleIter(object):
    def __init__(self, indices):
        self.indices = indices.copy()
        self.size = len(indices)
        self.pointer = self.size - 1
    
    def __next__(self):
        self.pointer += 1

        if self.pointer == self.size:
            self.pointer = 0
            random.shuffle(self.indices)
        
        return self.indices[self.pointer]
    
    def __iter__(self):
        return self
    

def balanced_generator(size, cls_iter, data_iters, repeat=1):
    """ 
    Arguments:
        @size: totoal number of samples 
        @cls_iter: class index iterator
        @data_iters: a list of per-class inde iterator 
        @repeat: number of samples for one class
    """
    cls_count = 0
    for _ in range(size):
        if cls_count % repeat == 0:
            cls_idx = next(cls_iter)
        cls_count = cls_count + 1 % repeat
        yield next(data_iters[cls_idx])


def chunk(iterable, chunk_size, drop_last):
    """ Convert index iterable to batch index iterable
        e.g. chunk([4, 2, 3, 1], 2) ==> [[4, 2], [3, 1]]
    """
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret and not drop_last:
        yield ret


class KVSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None,
                 shuffle=True, drop_last=False):
        super(KVSampler, self).__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        iterable = super(KVSampler, self).__iter__()
        return chunk(iterable, self.batch_size, self.drop_last)
    
    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class KVBalancedSampler(KVSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None,
                 shuffle=True, drop_last=False, repeat=1):
        super(KVBalancedSampler, self).__init__(
            dataset, batch_size, num_replicas=num_replicas, rank=rank,
            shuffle=shuffle, drop_last=drop_last)
        self.repeat = repeat
        self.percls_idxs = self.dataset.get_percls_idxs()
        self.cls_iter = RandomCycleIter(list(range(self.dataset.num_classes)))
        self.data_iters = [RandomCycleIter(idxs) for idxs in self.percls_idxs]

    def __iter__(self):
        iterable = balanced_generator(
            self.num_samples, self.cls_iter, self.data_iters, self.repeat)
        return chunk(iterable, self.batch_size, self.drop_last)


class DistributedBalancedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 repeat=1):
        super(DistributedBalancedSampler, self).__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.repeat = repeat
        self.percls_idxs = self.dataset.get_percls_idxs()
        self.cls_iter = RandomCycleIter(list(range(self.dataset.num_classes)))
        self.data_iters = [RandomCycleIter(idxs) for idxs in self.percls_idxs]

    def __iter__(self):
        return balanced_generator(
            self.num_samples, self.cls_iter, self.data_iters, self.repeat)
