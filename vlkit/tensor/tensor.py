import torch
import numpy as np

def numel(x):
    if isinstance(x, np.ndarray):
        return x.size
    elif isinstance(x, torch.Tensor):
        return x.numel()
    else:
        raise NotImplementedError("%s" % type(x))

