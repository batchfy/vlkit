import numpy as np
from ..array import isarray
from PIL import Image
from typing import Union


def get_input_shape(x: Union[np.ndarray, Image.Image, list], is_hw_first: bool = True):
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
        raise TypeError("Input must be a NumPy array, PyTorch tensor, an PIL.Image, or list of them.")
    return input_h, input_w


def crop_image(
    x: Union[np.ndarray, Image.Image],
    top: int, left: int, target_h: int, target_w: int, 
    is_hw_first: bool = True
    ) -> Union[np.ndarray, Image.Image]:
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
    return cropped


def random_crop(x: Union[np.ndarray, Image.Image, list], size: tuple, is_hw_first=True):
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
    input_h, input_w = get_input_shape(x, is_hw_first)

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
        x: Union[np.ndarray, Image.Image, list],
        mode:str='center',
        is_hw_first:bool=True
) -> Union[np.ndarray, Image.Image]:
    assert mode in ['center', 'random'], \
            f"mode must be either 'random' or 'center'."
    assert isarray(x) or isinstance(x, (Image.Image, list))
    input_h, input_w = get_input_shape(x, is_hw_first)
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


def center_crop(image: np.ndarray, target_size, is_hw_first:bool) -> np.ndarray:
    """
    Center crop an input image to the target size.
    If the target dimensions are larger than or equal to the input, the image is returned unchanged.

    Args:
        image (np.ndarray): Input image as a NumPy array (H x W x C or H x W).
        target_size (int or tuple): Target size as a single number (for square crop) or a tuple (target_h, target_w).
        is_hw_first (boolean): Are height and width the first two dimensions of the input.

    Returns:
        np.ndarray: Center-cropped image.
    """
    # Get input image dimensions
    if is_hw_first:
        input_h, input_w = image.shape[:2]
    else:
        input_h, input_w = image.shape[-2:]

    # Handle target_size input
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    # Check if cropping is necessary
    if target_h >= input_h and target_w >= input_w:
        return image

    # Compute cropping boundaries
    start_h = max((input_h - target_h) // 2, 0)
    end_h = start_h + target_h

    start_w = max((input_w - target_w) // 2, 0)
    end_w = start_w + target_w

    # Crop the image
    if is_hw_first:
        cropped_image = image[start_h:end_h, start_w:end_w, ]
    else:
        cropped_image = image[..., start_h:end_h, start_w:end_w]

    return cropped_image