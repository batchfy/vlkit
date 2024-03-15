from PIL import Image
import cv2, torch
import numpy as np
from . import __all_backends__
from .interpolation import get_interp, get_random_interp, __all_interpolations__


def format_size(size):
    assert isinstance(size, (int, list, tuple))
    if isinstance(size, (list, tuple)):
        assert len(size) == 2
    elif isinstance(size, int):
        size = (size, size)
    else:
        raise ValueError(size)
    return size


def resize(im, size, interpolation="bilinear", backend="pil"):
    assert isinstance(im, Image.Image)
    if backend == "random":
        backend = np.random.choice(__all_backends__, 1)[0]
    if interpolation == "random":
        interp = get_random_interp(backend=backend)
    else:
        interp = get_interp(interpolation, backend)

    h, w = format_size(size)
    if backend == "pil":
        im1 = im.resize((w, h), resample=interp)
    else:
        im1 = Image.fromarray(cv2.resize(np.array(im), (w, h), interpolation=interp))
    return im1


class Resize(torch.nn.Module):
    """Resize an image

    Args:
        size (int or tuple[int]): the target size
        interpolation (string, optional): interpolation, can be `random` or a specific interpolation method.
        backend (string, optional): the backend used to resize. Should be one of `cv2`, `pil` or `random`.
    """
    def __init__(self, size, interpolation="bilinear", backend="pil"):
        super().__init__()
        self.size = format_size(size)
        self.interpolation = interpolation
        self.backend = backend

    def forward(self, img):
        return resize(img, size=self.size, interpolation=self.interpolation, backend=self.backend)

    def __repr__(self):
	    return self.__class__.__name__ + '(size={0}, interpolation={1}, backend={2})'.format(
            self.size, self.interpolation, self.backend)
