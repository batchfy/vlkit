import torch.nn as nn
import math
from .conv import ConvModule

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, act_layer=nn.ReLU, act_args=None):
        super(SqueezeExcite, self).__init__()
        reduced_chs = math.ceil(se_ratio * in_chs)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_reduce_act = ConvModule(in_chs, reduced_chs, kernel_size=1, bias=True,
                act_layer=act_layer, act_args=act_args, norm_layer=None)

        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = self.pool(x)
        x_se = self.conv_reduce_act(x_se)
        x_se = self.conv_expand(x_se).expand_as(x)

        return x * self.sigmoid(x_se)
