try:
    import torch
except:
    torch = None
import numpy as np


def resample_volume(vol0, vol1):
    """
    resample volume-0 to align with volume-1
    """
    if not torch:
        raise ImportError(f"Torch is required.")
    min_coords0 = vol0.coords.reshape(-1, 3).min(axis=0)
    max_coords0 = vol0.coords.reshape(-1, 3).max(axis=0)
    size = max_coords0 - min_coords0
    normalized_coords1 = (vol1.coords - min_coords0) / size * 2 - 1
    normalized_coords1 = torch.tensor(normalized_coords1[None,])
    data = torch.tensor(vol0.values.astype(np.float64)[None, None,])
    return torch.nn.functional.grid_sample(data, normalized_coords1, align_corners=True).squeeze().numpy()
