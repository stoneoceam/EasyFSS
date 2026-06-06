r""" Helper functions """
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()


def is_distributed():
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def setup_distributed():
    if is_distributed():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    else:
        return False, 0


def move_to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        return sample.to(device, non_blocking=True)
    elif isinstance(sample, dict):
        return {k: move_to_device(v, device) for k, v in sample.items()}
    elif isinstance(sample, list):
        return [move_to_device(v, device) for v in sample]
    elif isinstance(sample, tuple):
        return tuple(move_to_device(v, device) for v in sample)
    else:
        return sample

def get_model_module(model):
    return model.module if hasattr(model, 'module') else model