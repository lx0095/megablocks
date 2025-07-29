# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

def sort(x, end_bit=None):
    return torch.sort(x)
