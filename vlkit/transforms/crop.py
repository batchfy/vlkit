import numpy as np


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


def center_crop(img, size:int):
    h, w = img.shape[:2]
    assert min(h, w) >= size, f"image shape ({h}x{w}) v.s. size {size}"
    return img[(h - size) // 2 : (h + size) // 2, (w - size) // 2 : (w + size) // 2, ]