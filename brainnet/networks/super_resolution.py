import torch

from brainnet.modules.blocks import ConvolutionBlock
from brainnet.modules.image import SubpixelConvolution


class SuperResolution(torch.nn.Sequential):
    def __init__(self, in_channels: int = 1, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)

        self.append(ConvolutionBlock(3, in_channels, 32, norm=False))
        self.append(ConvolutionBlock(3, 32, 32, norm=False))
        self.append(ConvolutionBlock(3, 32, 32, norm=False))
        self.append(ConvolutionBlock(3, 32, 32, norm=False))
        self.append(ConvolutionBlock(3, 32, 6, norm=False, activation=False))
        self.append(SubpixelConvolution(up_factor=6, up_dims=1, spatial_dims=3))
