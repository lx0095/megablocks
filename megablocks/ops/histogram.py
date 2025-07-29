# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

def histogram(x, bins):
    return torch.histogram(x, bins)