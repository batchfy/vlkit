import hashlib, cv2
import numpy as np
from .image import normalize


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


def overlay(image, mask, alpha=0.3, palette=None, show_boundary=False):
    """
    Overlay a mask on an image with optional boundary highlighting.

    Parameters:
    image (numpy.ndarray): The input image. Can be grayscale or RGB.
    mask (numpy.ndarray): The mask to overlay. Must have the same height and width as the image.
    alpha (float, optional): The transparency level of the mask overlay. Default is 0.3.
    palette (dict, optional): A dictionary mapping category indices to colors. If None, colors are generated.
    show_boundary (bool, optional): If True, boundaries of the mask regions are highlighted. Default is False.

    Returns:
    numpy.ndarray: The image with the mask overlay applied.
    """
    h, w = image.shape[:2]
    assert mask.shape == (h, w), f"Bad mask shape: {mask.shape}."
    if image.ndim == 2 or image.shape[-1] == 1:
        image = np.stack([image] * 3, axis=-1)
    assert image.shape[2] == 3, f"Bad image shape: {image.shape}."
    image = normalize(image, 0, 255).astype(np.uint8)
    overlay = image.copy()
    categories = np.unique(mask)
    palette = dict()
    for cat in categories:
        if cat != 0:
            palette[cat] = str2color(str(cat))
    for cat in palette:
        color = np.ones((h, w, 3)) * np.array(palette[cat])
        overlay1 = color * (mask == cat)[:, :, None] * alpha + image * (1 - alpha)
        overlay[mask == cat] = overlay1[mask == cat]
        if show_boundary:
            contours, _ = cv2.findContours((mask == cat).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return normalize(overlay, 0, 255).astype(np.uint8)

