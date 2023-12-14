import torch

import monai
from monai.networks.blocks import Convolution
from brainnet.modules.topofit.models import TopoFitGraph

"""
image -> feature extractor -> task nets -> prediction

Downstream task networks. These all takes as input a feature tensor

and returns a prediction.

"""

# def apply_postprocessing(module, *args, **kwargs):
#     return module.postprocess(*args, **kwargs) if hasattr(module, "postprocess") else


class TaskModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, extra_convs=None) -> None:
        super().__init__()

        extra_convs = [] if extra_convs is None else extra_convs
        assert isinstance(extra_convs, list)

        conv_kwargs = dict(spatial_dims=3)

        in_chs = [in_channels] + extra_convs
        out_chs = extra_convs + [out_channels]

        self.convs = torch.nn.Sequential(
            *[
                Convolution(in_channels=in_ch, out_channels=out_ch, **conv_kwargs)
                for in_ch, out_ch in zip(in_chs, out_chs)
            ]
        )

    def forward(self, features):
        return self.convs(features)


class SegmentationModule(TaskModule):
    def __init__(self, *args, dim=1, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, features):
        return self.softmax(super().forward(features))

    def postprocess(self, x, labels=None, dim=1):
        """Argmax with optional relabeling."""
        index = x.argmax(dim)
        return index if labels is None else labels[index]


class SuperResolutionModule(TaskModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class BiasFieldModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: torch.Size | torch.Tensor,
        out_shape: torch.Size | torch.Tensor,
    ) -> None:
        """Perform (up)convolution on the specified features and scale to the
        target shape if necessary.
        """
        super().__init__()

        # expose in config
        self.in_shape = torch.tensor(in_shape)
        self.out_shape = torch.tensor(out_shape)

        if self.in_shape.equal(self.out_shape):
            self.convolve = Convolution(in_channels, out_channels, spatial_dims=3)
        else:
            self.convolve = torch.nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=8, stride=8, padding=0
            )

        self.resize = monai.transforms.Resize(
            self.out_shape, mode="trilinear"
        )  # area is default

    def forward(self, features):
        if self.feature_level is not None:
            features = features[self.feature_level]
        x = self.convolve(features)
        x = x if self.out_shape.equal(torch.tensor(x.shape)) else monai.transforms.Resize(x)
        return x.exp()


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


class SurfaceModule(TopoFitGraph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
