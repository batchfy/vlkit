import numpy as np
try:
    import torch
except ImportError:
    torch = None


def normalize(x, lower_bound=0, upper_bound=255, eps=1e-6):
    """
    Normalize the input array or tensor to a specified range.

    This function adjusts the values in the input `x` to fall within the range
    specified by `lower_bound` and `upper_bound`. The normalization process
    ensures that the input's minimum and maximum values are scaled to align
    with the specified bounds. It supports both NumPy arrays and PyTorch tensors.

    Parameters:
    ----------
    x : np.ndarray or torch.Tensor
        The input array or tensor to be normalized. It should be a NumPy
        ndarray or a PyTorch Tensor. The input must have dimensionality between
        1 and 4.

    lower_bound : float, optional
        The lower bound of the normalized range. Default is 0.

    upper_bound : float, optional
        The upper bound of the normalized range. Default is 255.

    eps : float, optional
        A small constant to prevent division by zero when the maximum value
        of the input is zero. Default is 1e-6.

    Returns:
    -------
    np.ndarray or torch.Tensor
        The normalized array or tensor, reshaped to the original dimensions of `x`.

    Raises:
    ------
    RuntimeError
        - If `x` is neither a NumPy array nor a PyTorch tensor.
        - If `x` has a dimensionality greater than 4 or an unsupported shape.

    Notes:
    ------
    - For inputs with fewer than 4 dimensions, the normalization is applied 
      directly on the input.
    - For 4-dimensional inputs, the normalization is applied along the last
      axis while keeping batch-level structure intact.
    - PyTorch tensors and NumPy arrays are handled separately to leverage
      their respective backend operations.

    Example:
    -------
    >>> import numpy as np
    >>> x = np.array([[0, 50], [100, 150]])
    >>> normalize(x, lower_bound=0, upper_bound=1)
    array([[0.  , 0.333],
           [0.667, 1.   ]])
    """
    if isinstance(x, np.ndarray):
        backend = 'numpy'
    elif torch is not None and isinstance(x, torch.Tensor):
        backend = 'torch'
    else:
        raise RuntimeError(f"Input should be either a numpy.array" \
                "or torch.Tensor, but got {type(x)}")

    orig_shape = x.shape
    x = x.double() if backend == 'torch' else x.astype(np.float64)
    scale = upper_bound - lower_bound
    if x.ndim <= 3:
        x -= x.min()
        if x.max() > 0:
            x /= x.max()
    elif x.ndim == 4:
        x = x.reshape(x.shape[0], -1)
        if backend == 'torch':
            x -= x.min(dim=-1, keepdim=True).values
            x /= x.max(dim=-1, keepdim=True).values.clamp(min=eps)
        else:
            x -= x.min(axis=-1, keepdims=True)
            x /= np.clip(x.max(axis=-1, keepdims=True),
                    a_min=eps, a_max=None)
    else:
        raise RuntimeError('Invalid input shape: %s.' % str(x.shape))
    x *= scale
    x += lower_bound
    return x.reshape(*orig_shape)
