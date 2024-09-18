import torch

from brainnet.modules.blocks import ConvBlock
from brainnet.modules.topofit.models import TopoFitGraph, TopoFitGraphAdjust

"""
image -> feature extractor -> task nets -> prediction

Downstream task networks. These all takes as input a feature tensor

and returns a prediction.
1
"""

# def apply_postprocessing(module, *args, **kwargs):
#     return module.postprocess(*args, **kwargs) if hasattr(module, "postprocess") else


class HeadModule(torch.nn.Module):
    def __init__(self, channels: tuple | list[int], init_zeros: bool = False) -> None:
        """_summary_

        Parameters
        ----------
        channels : tuple | list[int]
            The first value defines number of input channels and the last value
            defines output channels. For example, [64, 64, 3] gives two conv
            layers: 64 -> 64 and 64 -> 3.
        """
        super().__init__()

        self.convs = torch.nn.Sequential(
            *[
                ConvBlock(3, in_ch, out_ch, init_zeros=init_zeros)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, features):
        return self.convs(features)


class SVFModule(torch.nn.Module):
    def __init__(self, channels: tuple | list[int], init_zeros: bool = True) -> None:
        super().__init__()

        izs = [False] * (len(channels) - 1)
        izs[-1] = True if init_zeros else False

        self.convs = torch.nn.Sequential(
            *[
                ConvBlock(3, in_ch, out_ch, init_zeros=iz)
                for in_ch, out_ch, iz in zip(channels[:-1], channels[1:], izs)
            ]
        )


    def forward(self, features):
        return self.convs(features)

class SegmentationModule(HeadModule):
    def __init__(self, *args, dim=1, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, features):
        return super().forward(features)

    def postprocess(self, x, labels=None, dim=1):
        """Argmax with optional relabeling."""
        # x = self.softmax(x)
        index = x.argmax(dim)
        return index if labels is None else labels[index]


class SuperResolutionModule(HeadModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


# class BiasFieldModule(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         in_shape: torch.Size | torch.Tensor,
#         out_shape: torch.Size | torch.Tensor,
#     ) -> None:
#         """Perform (up)convolution on the specified features and scale to the
#         target shape if necessary.
#         """
#         super().__init__()

#         # expose in config
#         self.in_shape = torch.tensor(in_shape)
#         self.out_shape = torch.tensor(out_shape)

#         if self.in_shape.equal(self.out_shape):
#             self.convolve = Convolution(in_channels, out_channels, spatial_dims=3)
#         else:
#             self.convolve = torch.nn.ConvTranspose3d(
#                 in_channels, out_channels, kernel_size=8, stride=8, padding=0
#             )

#         self.resize = monai.transforms.Resize(
#             self.out_shape, mode="trilinear"
#         )  # area is default

#     def forward(self, features):
#         if self.feature_level is not None:
#             features = features[self.feature_level]
#         x = self.convolve(features)
#         x = (
#             x
#             if self.out_shape.equal(torch.tensor(x.shape))
#             else monai.transforms.Resize(x)
#         )
#         return x.exp()


class ContrastiveModule(torch.nn.Module):
    def __init__(self, dim=1):
        """Normalize features.

        Parameters
        ----------
        dim : int, optional
            Dimension over which to normalize.

        References
        ----------
        https://openreview.net/forum?id=2oCb0q5TA4Y
        """
        super().__init__()

        self.dim = dim

    def forward(self, features):
        return torch.nn.functional.normalize(features, dim=self.dim)
        # return torch.nn.functional.normalize(features, dim=self.dim)

    # def forward(self, outputs, *kwargs):
    #     for output in outputs:
    #         output['feat'][-1] = F.normalize(output['feat'][-1], dim = 1)
    #     return outputs


SurfaceModule = TopoFitGraph
SurfaceAdjustModule = TopoFitGraphAdjust

surface_modules = (SurfaceModule, SurfaceAdjustModule)