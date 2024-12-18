import torch.nn as nn
from .conv import ConvModule
from .squeeze_excite import SqueezeExcite

class InvertedResidual(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, expand_ratio=1, se_ratio=None, act_layer=nn.ReLU):
        super(InvertedResidual, self).__init__()
        self.se = se_ratio is not None and se_ratio > 0
        self.residual = (in_chs == out_chs and stride == 1)
        hidden_dim = int(round(in_chs * expand_ratio))

        if act_layer == nn.ReLU:
            act_args = {"inplace": True}
        elif act_layer == nn.PReLU:
            act_args = {"num_parameters": -1}

        # point-wise convolution
        self.conv_pw = ConvModule(in_chs, hidden_dim, kernel_size=1,
                act_layer=act_layer, act_args=act_args, bias=False)

        # depth-wise convolution
        self.conv_dw = ConvModule(hidden_dim, hidden_dim, kernel_size=3,
                stride=stride, groups=hidden_dim,
                act_layer=act_layer, act_args=act_args, bias=False)

        # point-wise linear convolution
        self.conv_pwl = ConvModule(hidden_dim, out_chs, kernel_size=1, bias=False, act_layer=None)

        # se
        if self.se:
            self.se = SqueezeExcite(hidden_dim, se_ratio=se_ratio, act_layer=act_layer, act_args=act_args)

    def forward(self, x):
        residual = x
        x = self.conv_pw(x)
        x = self.conv_dw(x)

        if self.se:
            x = self.se(x)

        x = self.conv_pwl(x)

        if self.residual:
            x += residual

        return x

