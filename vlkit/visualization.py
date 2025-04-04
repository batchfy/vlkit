import hashlib
import numpy as np
from io import BytesIO
from PIL import Image
from einops import rearrange

from .array import torch, cv2
from .image import normalize
from .image.format import convert2hw3


def iscolor(x):
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 3:
        return all(0 <= v <= 255 for v in x)
    return False


def str2color(s: str) -> tuple:
    """
    Maps a string to a consistent RGB color.

    Args:
        s (str): The input string.

    Returns:
        tuple: A tuple (r, g, b) with each value in the range [0, 255].
    """
    hash_object = hashlib.sha256(s.encode('utf-8'))
    hex_digest = hash_object.hexdigest()

    r = int(hex_digest[0:2], 16)
    g = int(hex_digest[2:4], 16)
    b = int(hex_digest[4:6], 16)

    return (r, g, b)


def overlay_mask(image, mask, alpha=0.3, palette=None, show_boundary=False, boundary_color=(255, 255, 255)):
    """
    Overlay a categorical mask on an image with optional boundary highlighting.

    Parameters:
    image (numpy.ndarray): The input image. Can be grayscale or RGB.
    mask (numpy.ndarray): The mask to overlay. Must have the same height and width as the image.
    alpha (float, optional): The transparency level of the mask overlay. Default is 0.3.
    palette (dict, optional): A dictionary mapping category indices to colors. If None, colors are generated.
    show_boundary (bool, optional): If True, boundaries of the mask regions are highlighted. Default is False.

    Returns:
    numpy.ndarray: The image with the mask overlay applied.
    """
    if cv2 is None:
        raise ImportError(
            "cv2 is required for this function. Install via pip 'install opencv-python'."
        )
    h, w = image.shape[:2]
    assert mask.shape == (h, w), f"Bad mask shape: {mask.shape}."

    mask = mask.astype(np.uint8)
    if image.ndim == 2 or image.shape[-1] == 1:
        image = np.stack([image] * 3, axis=-1)
    assert image.shape[2] == 3, f"Bad image shape: {image.shape}."

    image = normalize(image, 0, 255).astype(np.uint8)
    overlay = image.copy()

    categories = np.unique(mask)

    if palette is None:
        palette = dict()
    for cat in categories:
        if cat != 0:
            if cat not in palette:
                palette[cat] = str2color(str(cat))
            assert iscolor(palette[cat]), f"Bad color for category {cat}: {palette[cat]}."

    for cat in palette:
        color = np.ones((h, w, 3)) * np.array(palette[cat])
        overlay1 = color * (mask == cat)[:, :, None] * alpha + image * (1 - alpha)
        overlay[mask == cat] = overlay1[mask == cat]
        if show_boundary:
            assert len(boundary_color) == 3, f"Bad boundary color: {boundary_color}."
            contours, _ = cv2.findContours((mask == cat).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, boundary_color, 1)
    return normalize(overlay, 0, 255).astype(np.uint8)


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET, threshold=0.0):
    """
    Overlay a heatmap on an image with threshold control.

    Args:
        image (numpy.ndarray): The original image (H, W, 3).
        heatmap (numpy.ndarray): The heatmap with values in range [0,1] (H, W).
        alpha (float): Transparency factor (0: only image, 1: only heatmap).
        colormap (int): OpenCV colormap type.
        threshold (float): Values below this threshold will not be overlaid (range [0,1]).

    Returns:
        numpy.ndarray: Image with heatmap overlay.
    """
    if cv2 is None:
        raise ImportError(
            "cv2 is required for this function. Install via pip 'install opencv-python'."
        )
    assert heatmap.ndim == 2, f"Bad heatmap shape: {heatmap.shape}."
    assert heatmap.min() >= 0 and heatmap.max() <= 1, f"Bad heatmap range: [{heatmap.min()}, {heatmap.max()}]." 
    
    image = convert2hw3(image)
    
    # Convert image to uint8 if needed
    if image.dtype != np.uint8:
        image = np.uint8(normalize(image, 0, 1) * 255)
    
    # Create mask for values above threshold
    mask = heatmap >= threshold
    
    # Normalize heatmap to range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Apply color map
    heatmap_colored = cv2.cvtColor(cv2.applyColorMap(heatmap, colormap), cv2.COLOR_BGR2RGB)
    
    # Initialize overlay with original image
    overlay = image.copy()
    
    # Only blend where mask is True
    overlay[mask] = cv2.addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)[mask]

    return overlay


def plot2array(fig, order="nchw", dpi=100):
    """Convert a Matplotlib figure to a TensorBoard-compatible image (C, H, W)."""
    if torch is None:
        raise ImportError("torch is required for this function.")
    assert order in ["nchw", "hwc"], f"Invalid order: {order}."

    buf = BytesIO()
    fig.savefig(buf, format='jpg', dpi=dpi)  # Save figure to buffer
    buf.seek(0)

    # Open image with PIL and convert to NumPy array
    image = Image.open(buf)
    image = np.array(image)  # Shape (H, W, 3) with RGB

    if order == "nchw":
        image = rearrange(image, "h w c -> 1 c h w")

    return image