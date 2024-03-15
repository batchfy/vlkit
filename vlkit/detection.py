import numpy as np
from PIL import Image, ImageDraw

def visualize_bboxes(image, bboxes, width=2, color="red"):
    assert isinstance(image, np.ndarray) and isinstance(bboxes, np.ndarray)
    if bboxes.ndim == 1:
        assert bboxes.size == 4
        bboxes = bboxes.reshape((1,4))
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4, "%s" % bboxes.shape
    if isinstance(color, str):
        color = [color] * bboxes.shape[0]
    assert len(color) == bboxes.shape[0]

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for idx, b in enumerate(bboxes):
        draw.rectangle(b.tolist(), width=width, outline=color[idx])

    return np.array(image)

def bbox2mask(bbox, size):
    """
    convert bounding box to binary mask
    bbox: a bounding box represeting [x1, y1, x2, y2]
    size: size of the mask [H, W]
    """
    x1, y1, x2, y2 = bbox
    mask = np.zeros(size, dtype=bool)
    ys, xs = np.meshgrid(np.arange(y1, y2),
                      np.arange(x1, x2),
                      indexing='ij')
    mask[ys.astype(int), xs.astype(int)] = True

    return mask

def box_coverage(bbox, mask):
    """
    coverage of a box w.r.t the GT mask
    bbox: a bounding box represeting [x1, y1, x2, y2]
    mask: the GT binary mask
    """
    assert mask.ndim == 2
    assert bbox.size == 4
    bbox_mask = bbox2mask(bbox, mask.shape)
    assert bbox_mask.shape == mask.shape
    return np.logical_and(bbox_mask, mask).sum() / max(mask.sum(), 1)


