try:
    import torch
except Exception as e:
    torch = None

try:
    import numpy as np
except Exception as e:
    numpy = np = None

try:
    import cv2
except Exception as e:
    cv2 = None

from PIL import Image

if np == None and torch == None:
    raise ImportError(f"At least one of torch and numpy should be installed.")

def isarray(x):
    if torch is not None:
        return isinstance(x, (np.ndarray, torch.Tensor))
    else:
        return isinstance(x, np.ndarray)

def array2type(arr, to_type):
    if type(arr) == to_type:
        return arr
    if type(arr) ==  Image.Image and  to_type == np.ndarray:
        return np.array(arr)
    if type(arr) == np.ndarray and to_type == Image.Image:
        return Image.fromarray(arr)
    if torch is not None:
        if type(arr) == torch.Tensor and to_type == np.ndarray:
            return arr.cpu().detach().numpy()