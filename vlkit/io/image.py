import torch
import numpy as np
from PIL import Image
import os.path as osp


def read2tensor(fn):
    assert osp.isfile(fn)
    im = np.array(Image.open(fn)).transpose((2, 0, 1))
    return torch.tensor(im).unsqueeze(dim=0)