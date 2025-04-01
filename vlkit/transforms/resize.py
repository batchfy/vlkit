from PIL import Image
import cv2, torch
import numpy as np
from ..array import array2type
from . import __all_backends__
from .interpolation import get_interp, get_random_interp, __all_interpolations__


def resize_short_edge(im, size, interpolation="bilinear", backend="pil"):
    """Resize image to ensure the shortest edge equals the given size.
    Aspect ratio is preserved.
    
    Args:
        im (PIL.Image): Input image
        size (int): Target size for shortest edge
        interpolation (str): Interpolation method
        backend (str): Backend for resizing
    """
    assert isinstance(im, (Image.Image, np.ndarray)), type(im)
    if isinstance(im, np.ndarray):
        h, w = im.shape[:2]
    elif isinstance(im, Image.Image):
        w, h = im.size
    else:
        raise TypeError(type(im))

    scale = size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return resize(im, (new_h, new_w), interpolation, backend)


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
    assert isinstance(im, (Image.Image, np.ndarray))
    input_type = type(im)
    if backend == "random":
        backend = np.random.choice(__all_backends__, 1)[0]

    if backend == "pil" and isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    elif backend == "cv2" and isinstance(im, Image.Image):
        im = np.array(im)

    if interpolation == "random":
        interp = get_random_interp(backend=backend)
    else:
        interp = get_interp(interpolation, backend)

    h, w = format_size(size)
    if backend == "pil":
        im1 = im.resize((w, h), resample=interp)
    elif backend == "cv2":
        im1 = cv2.resize(im, (w, h), interpolation=interp)
    else:
        raise ValueError(backend)

    return array2type(im1, input_type)


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
