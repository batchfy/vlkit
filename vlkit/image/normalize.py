import numpy as np
try:
    import torch
except ImportError:
    torch = None


def normalize(x, lower_bound=0, upper_bound=255, eps=1e-6):
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
