import torch


class Upsample2d(torch.nn.Module):
    def __init__(self, in_channels, channels, factor=2) -> None:
        super().__init__()
        self.channels = channels
        self.factor = factor
        self.conv3x3 = torch.nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False)

    def forward(self, input):
        return torch.nn.functional.interpolate(
            self.conv3x3(input),
            scale_factor=self.factor,
            mode='bilinear',
            align_corners=True
        )