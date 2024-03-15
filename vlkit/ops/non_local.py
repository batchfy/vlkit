import torch
import torch.nn as nn
from einops import rearrange

class NonLocal(torch.nn.Module):
    def __init__(self, in_chs, hidden_chs=None, return_affinity=False):
        super().__init__()
        self.in_chs = in_chs
        if hidden_chs is not None:
            self.hidden_chs = hidden_chs
        else:
            self.hidden_chs = in_chs
        self.return_affinity = return_affinity

        self.conv_k = nn.Conv2d(self.in_chs, self.hidden_chs, kernel_size=1, bias=False)
        self.conv_q = nn.Conv2d(self.in_chs, self.hidden_chs, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(self.in_chs, self.hidden_chs, kernel_size=1, bias=False)

        self.conv = nn.Conv2d(self.hidden_chs, self.in_chs, kernel_size=1, bias=False)

    def forward(self, x):
        assert x.ndim == 4
        n, c, h, w = x.shape

        k = self.conv_k(x)
        q = self.conv_q(x)
        v = self.conv_v(x)

        k = rearrange(k, "n c h w -> n (h w) c")
        q = rearrange(q, "n c h w -> n c (h w)")
        v = rearrange(v, "n c h w -> n c (h w)")

        affinity = torch.bmm(k, q).softmax(dim=0)
        x = torch.bmm(v, affinity).view(n, -1, h, w)
        v = v.view(n, -1, h, w)
        x = self.conv(x + v)

        if self.return_affinity:
            return x, affinity
        else:
            return x
