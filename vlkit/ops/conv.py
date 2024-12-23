import torch


def conv1d(u, v):
    """
    Convolve each row of matrix u and v
    """
    assert u.ndim == v.ndim == 2
    assert u.shape[0] == v.shape[0]
    n = u.shape[0]
    d1, d2 = u.shape[1], v.shape[1]
    u = u.view(1, n, d1)
    v = v.view(n, 1, d2).flip(dims=(-1,))
    return torch.nn.functional.conv1d(u, v, padding=d2-1, groups=n).squeeze(dim=0)


if __name__ == '__main__':
    u = torch.ones(10, 5)
    v = torch.tensor([1, 2, 1]).view(1, 3).repeat(10, 1).to(u)
    print(u.shape, v.shape)
    print(conv1d(u, v), conv1d(u, v).shape)