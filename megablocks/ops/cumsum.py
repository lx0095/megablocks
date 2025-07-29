# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

def exclusive_cumsum(x, dim):
    return x.cumsum(dim) - x

def inclusive_cumsum(x, dim):
    return x.cumsum(dim)
