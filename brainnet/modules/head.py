import torch

from brainnet.modules.blocks import ConvBlock
# from brainnet.modules.cortexnet import CortexThing
from brainnet.modules.topofit import TopoFit#, TopoReg

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
    def __init__(self, channels: tuple | list[int], feature_maps: list[str], init_zeros: bool = False) -> None:
        super().__init__()
        self.feature_maps = feature_maps

        izs = [False] * (len(channels) - 1)
        izs[-1] = init_zeros

        self.convs = torch.nn.Sequential(
            *[
                ConvBlock(3, in_ch, out_ch, init_zeros=iz)
                for in_ch, out_ch, iz in zip(channels[:-1], channels[1:], izs)
            ]
        )

    def cat_features(self, features):
        return torch.cat([features[m] for m in self.feature_maps], dim=1)

    def forward(self, features):
        return self.convs(self.cat_features(features))

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


# surface_modules = (CortexThing, TopoFit)
surface_modules = (TopoFit, )

