import torch


class DropPath(torch.nn.Module):
    """
    Randomly drop paths (zero output) per sample.
    p: probability of dropping samples
    """
    def __init__(self, p=0):
        super().__init__()
        assert 0 <= p < 1
        self.p = p

    def forward(self, x):
        if self.p == 0 or self.training:
            return x
        drop = torch.rand(x.shape[0]).to(x) < self.p
        drop_shape = [x.shape[0]] + [1,] * (x.ndim - 1)
        drop = drop.view(drop_shape)

        return x.div(1 - self.p) * torch.logical_not(drop)

    def extra_repr(self):
        return ("p={p}".format(p=self.p))
