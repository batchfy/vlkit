import torch
import numpy as np
from ..utils import isarray


def minmax_normalize(x):
    if isarray(x, np.ndarray):
        x -= x.min()
        x /= x.max()
    else:
        raise TypeError('invalid input type %s' % type(x))
    return x


if __name__ == "__main__":
    minmax_normalize(torch.zeros(1, 4))
