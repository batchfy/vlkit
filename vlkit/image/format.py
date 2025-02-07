from ..array import torch
import numpy as np


def convert2hw3(image: np.ndarray):
    assert image.ndim == 2 or image.ndim == 3, f"Bad image shape: {image.shape}."

    if image.ndim == 3:
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=2)
        elif image.shape[2] != 3:
            raise ValueError(f"Bad image shape: {image.shape}.")

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    return image
