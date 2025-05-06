import numpy as np
from scipy.ndimage import distance_transform_edt
from ..common import Dotdict


def bwdist(mask):
    """distance transform, alternative to Matlab 'bwdist' <https://www.mathworks.com/help/images/ref/bwdist.html>.
    Computes the distance transform of a binary mask, similar to MATLAB's 'bwdist' function.

    This function calculates the Euclidean distance from each zero-valued pixel in the input mask 
    to the nearest non-zero pixel. It also returns the indices of the nearest non-zero pixels 
    and the offset from the original pixel positions.

    Parameters:
    -----------
    mask : numpy.ndarray
        A 4D binary mask of shape (n, c, h, w), where:
        - n is the batch size,
        - c is the number of channels (must be 1),
        - h and w are the height and width of the mask.
        The mask should contain boolean or binary values (0 or 1).

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - 'offset': numpy.ndarray of shape (n, 2, h, w)
          The offset from each pixel to the nearest non-zero pixel, in (y, x) order.
        - 'distance': numpy.ndarray of shape (n, c, h, w)
          The Euclidean distance from each pixel to the nearest non-zero pixel.
        - 'indices': numpy.ndarray of shape (n, 2, h, w)
          The indices of the nearest non-zero pixel for each pixel, in (y, x) order.

    Notes:
    ------
    - The input mask must have exactly one channel (c = 1).
    - The function uses `scipy.ndimage.distance_transform_edt` to compute the distance transform 
      and the indices of the nearest non-zero pixels.

    Example:
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import distance_transform_edt
    >>> mask = np.array([[[[0, 1], [1, 0]]]], dtype=bool)
    >>> result = bwdist(mask)
    >>> print(result['distance'])
    >>> print(result['offset'])
    >>> print(result['indices'])
    
    """
    assert mask.ndim == 4
    assert np.unique(mask).size <= 2
    mask = mask.astype(bool)
    n, c, h, w = mask.shape
    assert c == 1
    distance = np.full(mask.shape, np.inf, dtype=np.float32)
    indices = np.full((n, 2, h, w), -1, dtype=np.int32)
    offset = np.full(indices.shape, np.inf, dtype=np.float32)
    yxs = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), axis=0)
    for i in range(n):
        if mask[i].sum() > 0:
            # ind: [2 h w] in y, x order.
            dist, ind = distance_transform_edt(~mask[i, 0], return_indices=True)
            indices[i] = ind
            offset[i] = ind - yxs
            distance[i, 0] = dist
    return Dotdict(offset=offset, distance=distance, indices=indices)
