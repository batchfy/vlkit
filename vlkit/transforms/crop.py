import numpy as np
from ..array import isarray
from PIL import Image, ImageOps
from typing import Union


ImgOrList = Union[np.ndarray, Image.Image, list]
ArrayOrImage = Union[np.ndarray, Image.Image]


def get_input_hw(x: ImgOrList, is_hw_first: bool = True):
    """
    Get the input image shape.

    Parameters:
    x (Union[np.ndarray, torch.Tensor, Image.Image, list]): The input image(s). It can be a NumPy array, 
        PyTorch tensor, PIL.Image, or a list of these types.
    is_hw_first (bool, optional): If True, the shape is returned as (height, width). If False, the shape 
        is returned as (width, height). Default is True.

    Returns:
    tuple: A tuple containing the height and width of the input image(s).

    Raises:
    TypeError: If the input is not a NumPy array, PyTorch tensor, PIL.Image, or list of these types.
    AssertionError: If the input is a list of images and they do not all have the same size.
    """
    if isinstance(x, Image.Image):
        input_w, input_h = x.size
    elif isarray(x):
        input_h, input_w = (x.shape[:2] if is_hw_first else x.shape[-2:])
    elif isinstance(x, list):
        assert len(x) > 0
        if isarray(x[0]):
            input_h, input_w = (x[0].shape[:2] if is_hw_first else x[0].shape[-2:])
            for x1 in x:
                assert (input_h, input_w) == x1.shape[:2]
        elif isinstance(x[0], Image.Image):
            input_w, input_h = x[0].size
            for x1 in x:
                assert (input_w, input_h) == x1.size, "All images must have the same size."
        else:
            raise TypeError("Input list must contain NumPy arrays or PIL.Images.")
    else:
        raise TypeError(f"Input must be a NumPy array, PyTorch tensor, an PIL.Image, or list of them, but got {type(x)}.")
    return input_h, input_w


def crop_image(
    x: ArrayOrImage,
    top: int, left: int, target_h: int, target_w: int, 
    is_hw_first: bool = True
    ) -> ArrayOrImage:
    """
    Crop an image or an array to the specified dimensions.

    Parameters:
    x (PIL.Image.Image or numpy.ndarray): The input image or array to be cropped.
    top (int): The top pixel coordinate for the crop.
    left (int): The left pixel coordinate for the crop.
    target_h (int): The height of the cropped area.
    target_w (int): The width of the cropped area.
    is_hw_first (bool, optional): If True, the array is assumed to have the shape (height, width, ...).
                  If False, the array is assumed to have the shape (..., height, width).
                  Default is True.

    Returns:
    PIL.Image.Image or numpy.ndarray: The cropped image or array.
    """
    if isinstance(x, Image.Image):
        cropped = x.crop((left, top, left + target_w, top + target_h))
    elif isarray(x):
        if is_hw_first:
            cropped = x[top:top + target_h, left:left + target_w, ...]
        else:
            cropped = x[..., top:top + target_h, left:left + target_w]
    else:
        raise TypeError("Input must be a NumPy array or a PIL.Image.")
    return cropped



def random_crop(x: ImgOrList, size: tuple, is_hw_first=True):
    """
    Perform a random crop on the input tensor.
    
    Parameters:
    ----------
    x : np.ndarray or torch.Tensor or PIL.Image or list of them.
    size : tuple
        Target crop size as (h, w).
    is_hw_first : bool
        Indicates if the height and width are the first dimensions (True) 
        or the last dimensions (False).
        
    Returns:
    -------
    np.ndarray or torch.Tensor
        Cropped tensor with the same type as input.
    """
    assert isarray(x) or isinstance(x, (Image.Image, list))

    target_h, target_w = size
    input_h, input_w = get_input_hw(x, is_hw_first)

    if target_h > input_h or target_w > input_w:
        raise ValueError("Target size cannot be larger than the input size.")
    
    # Generate random starting points for cropping
    top = np.random.randint(0, input_h - target_h + 1)
    left = np.random.randint(0, input_w - target_w + 1)

    if isinstance(x, list):
        cropped = [crop_image(x1, top, left, target_h, target_w, is_hw_first) for x1 in x]
    else:
        cropped = crop_image(x, top, left, target_h, target_w, is_hw_first)
    return cropped


def crop_long_edge(
        x: ImgOrList,
        mode:str='center',
        is_hw_first:bool=True
) -> ImgOrList:
    assert mode in ['center', 'random'], \
            f"mode must be either 'random' or 'center'."
    assert isarray(x) or isinstance(x, (Image.Image, list))
    input_h, input_w = get_input_hw(x, is_hw_first)
    crop_size = min(input_h, input_w)
    if mode == 'random':
        top = 0 if input_h == crop_size else \
            np.random.randint(0, input_h - crop_size)
        left = 0 if input_w == crop_size else \
            np.random.randint(0, input_w - crop_size)
    else:
        top = 0 if input_h == crop_size else \
            int((input_h - crop_size) // 2)
        left = 0 if input_w == crop_size else \
            int((input_w - crop_size) // 2)
    if isinstance(x, list):
        cropped = [crop_image(x1, top, left, crop_size, crop_size, is_hw_first) for x1 in x]
    else:
        cropped = crop_image(x, top, left, crop_size, crop_size, is_hw_first)
    return cropped


def center_crop(x: ImgOrList, target_size: tuple | int, is_hw_first:bool) -> ImgOrList:
    # Get input image dimensions
    input_h, input_w = get_input_hw(x, is_hw_first)

    # Handle target_size input
    assert isinstance(target_size) or len(target_size) == 2, f"Invalid target_size: {target_size}."
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    # Check if cropping is necessary
    if target_h >= input_h and target_w >= input_w:
        return x

    # Compute cropping boundaries
    start_h = max((input_h - target_h) // 2, 0)
    end_h = start_h + target_h

    start_w = max((input_w - target_w) // 2, 0)
    end_w = start_w + target_w

    # Crop the image
    if isinstance(x, list):
        cropped = [crop_image(x1, start_h, start_w, target_h, target_w, is_hw_first) for x1 in x]
    else:
        cropped = crop_image(x, start_h, start_w, target_h, target_w, is_hw_first)
    return cropped


def crop_to_divisible(
    x: ImgOrList,
    divisor: int, is_hw_first: bool = True
) -> ImgOrList:
    """
    Crop an input image to make its height and width divisible by a given number.

    Parameters:
    x (np.ndarray): Input image of shape [h, w, c] or [c, h, w] if is_hw_first is False, 
                    otherwise [c, h, w] or [h, w].
    divisor (int): The number by which the height and width should be divisible.
    is_hw_first (bool, optional): If True, the array is assumed to have the shape (height, width, ...).
                                  If False, the array is assumed to have the shape (..., height, width).
                                  Default is True.

    Returns:
    np.ndarray: The cropped image.
    """
    h, w = get_input_hw(x, is_hw_first)

    new_h = int(np.floor(h / divisor) * divisor)
    new_w = int(np.floor(w / divisor) * divisor)

    if new_h == h and new_w == w:
        return x
    if isinstance(x, list):
        divisible = [crop_to_divisible(x1, divisor, is_hw_first) for x1 in x]
    else:
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        divisible = crop_image(x, start_h, start_w, new_h, new_w, is_hw_first)
    return divisible