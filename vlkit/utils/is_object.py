from typing import Any
import numpy as np
from PIL import Image
import os.path as osp

from ..array import torch


__all_img_exts__ = ['.png', '.jpeg', '.jpg', '.bmp']


def isarray(a: Any) -> bool:
    return isinstance(a, (torch.Tensor, np.ndarray))


def isimg(fn: str) -> bool:
    return osp.splitext(fn)[-1].lower() in __all_img_exts__

