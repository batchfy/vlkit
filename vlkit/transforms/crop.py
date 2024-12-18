import numpy as np
try:
    import torch
except:
    torch = None

def random_crop(x, size: tuple, is_hw_first=True):
    """
    Perform a random crop on the input tensor.
    
    Parameters:
    ----------
    x : np.ndarray or torch.Tensor
        Input tensor of shape [h, w, ...] or [..., h, w].
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
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise ValueError("Input x must be a numpy.ndarray or torch.Tensor.")
    
    input_h, input_w = (x.shape[:2] if is_hw_first else x.shape[-2:])
    target_h, target_w = size

    if target_h > input_h or target_w > input_w:
        raise ValueError("Target size cannot be larger than the input size.")
    
    # Generate random starting points for cropping
    top = np.random.randint(0, input_h - target_h + 1)
    left = np.random.randint(0, input_w - target_w + 1)
    
    if is_hw_first:
        cropped = x[top:top + target_h, left:left + target_w, ...]
    else:
        cropped = x[..., top:top + target_h, left:left + target_w]
    
    return cropped

def crop_long_edge(img, mode:str='center'):
    assert mode in ['center', 'random'], \
            f"mode must be either 'random' or 'center'."
    img_height, img_width = img.shape[:2]
    crop_size = min(img_height, img_width)
    if mode == 'random':
        y1 = 0 if img_height == crop_size else \
            np.random.randint(0, img_height - crop_size)
        x1 = 0 if img_width == crop_size else \
            np.random.randint(0, img_width - crop_size)
    else:
        y1 = 0 if img_height == crop_size else \
            int((img_height - crop_size) // 2)
        x1 = 0 if img_width == crop_size else \
            int((img_width - crop_size) // 2)
    y2, x2 = y1 + crop_size, x1 + crop_size
    return img[y1:y2, x1:x2]


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