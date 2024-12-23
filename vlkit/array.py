try:
    import torch
except Exception as e:
    torch = None

try:
    import numpy as np
except Exception as e:
    numpy = np = None


if np == None and torch == None:
    raise ImportError(f"At least one of torch and numpy should be installed.")