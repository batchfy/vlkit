import torch
import torch.nn as nn
from vlkit.ops import ConvModule


class DualBN(nn.Module):
    def __init__(self, in_chs, num_bns):
        super(DualBN, self).__init__()
        assert num_bns > 1 and in_chs > 0
        self.num_bns = num_bns
        self.bns = nn.ModuleList([nn.BatchNorm2d(in_chs) for _ in range(num_bns)])

    def forward(self, x, weights):
        """
        x: input tensor with shape [N C H W]
        weights: bn weights with possible shapes: ([N num_bns] | [N] | [1])
        """
        bs = x.shape[0]
        assert isinstance(weights, (torch.Tensor, int))

        if isinstance(weights, int) or weights.numel() == 1:
            weights = weights.item() if isinstance(weights, torch.Tensor) else weights
            output = self.bns[weights](x)

        elif weights.numel() == bs * self.num_bns:
            """
            weights: [bs x num_bns]
            weighted average
            """
            assert weights.numel() == self.num_bns or weights.numel() ==  self.num_bns * bs, \
                    "weights.shape=%s v.s. self.num_bns=%d" % (weights.shape, self.num_bns)
            output = torch.cat([bn(x).unsqueeze(dim=0) for bn in self.bns], dim=0)
            output = (output * weights.view(self.num_bns, -1, 1, 1, 1)).sum(dim=0)

        elif weights.numel() == bs:
            assert weights.max() < self.num_bns
            output = torch.zeros_like(x)
            for i in range(self.num_bns):
                indice = weights == i
                if indice.sum() == 0:
                    continue
                if indice.sum() == 1:
                    raise ValueError("single sample for BN-%d us not permitted" % i)

                if indice.sum() != 0:
                    output[indice] = self.bns[i](x[indice])
        return output


class ConvDualBNReLU(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1, num_bns=5):
        super(ConvDualBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=False)
        self.bn = DualBN(out_chs, num_bns)
        self.relu = nn.PReLU(out_chs)

    def forward(self, x, bn_weights):
        x = self.conv(x)
        x = self.bn(x, bn_weights)
        x = self.relu(x)
        return x
