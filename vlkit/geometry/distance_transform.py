import numpy as np
from scipy.ndimage import distance_transform_edt


def bwdist(x):
    """distance transform, alternative to Matlab 'bwdist' <https://www.mathworks.com/help/images/ref/bwdist.html>.

    Args:
        x (np.ndarray): a [h w] binary map

    Returns:
        (tuple): tuple containing:

            dist (np.ndarray): the distance matrix [h w] \n
            yxs (np.ndarray): the coordinates of nearest points [2 h w] \n
            field (np.ndarray): the directional vector field [2 h w]

    .. image:: _static/distance_transform.svg

    See `an example <https://github.com/vlkit/vlkit/blob/master/examples/distance_transform.ipynb>`_ here 
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    h, w = x.shape

    x = x.astype(bool)
    dist, yxs = distance_transform_edt(np.logical_not(x), return_distances=True, return_indices=True)

    ys = np.arange(h).reshape(-1, 1)
    xs = np.arange(w).reshape(1, -1)
    field = yxs - np.stack((np.tile(ys, (1, w)), np.tile(xs, (h, 1))))
    return dist, yxs, field


def batch_bwdist(x):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3
    n, h, w = x.shape

    dist = np.zeros_like(x, dtype=np.float64)
    yxs = np.zeros((n, 2, h, w), dtype=int)
    field = np.zeros((n, 2, h, w), dtype=np.float64)

    for i in range(n):
        d1, yx1, f1 = bwdist(x[i,])
        dist[i, ] = d1
        yxs[i, ] = yx1
        field[i, ] = f1
    return dist, yxs, field
