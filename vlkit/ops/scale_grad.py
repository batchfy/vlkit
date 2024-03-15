import torch


class ScaleGradFunc(torch.autograd.Function):
    """
    Scale the gradient
    """
    @staticmethod
    def forward(ctx, x, scale=1):
        ctx.scale = scale
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


scale_grad = ScaleGradFunc.apply


class ScaleGrad(torch.nn.Module):
    """
    Scale gradient.
    This module can be used to inverse the gradient (e.g. scale=-1)
    or block the gradient (e.g. scale=0)
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return scale_grad(x, self.scale)

