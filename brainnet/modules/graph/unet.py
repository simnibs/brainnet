from typing import Type

import torch

import brainnet.mesh.topology
from brainnet.modules.graph import layers


def make_unet_channels(in_channels: int, depth: int, multiplier: int = 2) -> dict:
    """Construct Unet hierarchy"""

    assert depth >= 1
    m = depth - 1
    # encoder = [in_channels * multiplier**i for i in range(m)]
    # ubend = in_channels * multiplier**m
    # return dict(encoder=encoder, ubend=ubend, decoder=decoder)
    encoder = [in_channels * multiplier**i for i in range(m + 1)]
    decoder = encoder[:-1][::-1]
    return dict(encoder=encoder, decoder=decoder)


class EncoderUnit(torch.nn.Module):
    def __init__(
        self,
        conv_kwargs: dict,
        pool_kwargs: dict,
        do_pool: bool = True,
    ):
        super().__init__()

        self.do_pool = do_pool
        self.conv = torch.nn.Sequential(
            layers.ConvolutionRepeater(**conv_kwargs),
        )
        if self.do_pool:
            self.pool = layers.Pool(**pool_kwargs)

    def forward(self, features):
        features = self.conv(features)
        skip_features = features
        features = self.pool(features) if self.do_pool else features
        return features, skip_features


class Concatenate(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *args):
        return torch.concat(args, dim=self.dim)


class DecoderUnit(torch.nn.Module):
    def __init__(
        self,
        conv_kwargs: dict,
        unpool_kwargs: dict,
    ):
        super().__init__()
        self.unpool = layers.Unpool(**unpool_kwargs)
        self.concatenate = Concatenate(dim=1)
        self.conv = layers.ConvolutionRepeater(**conv_kwargs)

    def forward(self, features, skip_features):
        return self.conv(self.concatenate(self.unpool(features), skip_features))


class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        topologies: list[brainnet.mesh.topology.Topology],
        conv_module: Type[torch.nn.Module],
        reduce: str = "amax",
        channels: int | dict = 32,
        max_depth: int = 4,
        n_conv: int = 1,
    ):
        """Similar to a conventional UNet achitecture but on a graph. We
        exploit the fact that the topologies represent a hierarchy of
        recursive subdivision. Consequently, we can move up and down this
        hierarchy to obtain different mesh resolutions.


        Parameters
        ----------
        in_channels : int
        topologies :
        conv_module
        reduce: str
        channels: int | dict
        max_depth: int
        multiplier: int
        n_conv: int


        """
        # The UNet architecture and naming
        #
        # ENCODER                         DECODER   Hierarchy level (example)
        #
        # I C C ------------------------- I C C     4
        #     P                           U
        #     I C C ----------------- I C C         3
        #         P                   U
        #         I C C --------- I C C             2
        #             P           U
        #             I C C - I C C                 1
        #                 P   U
        #                 I C C                     0
        #                 U-bend
        #
        # I : input
        # C : conv
        # P : pooling
        # U : unpooling
        # - : skip connection
        #
        # Encoder unit: (I-)C-C-P
        # Decoder unit: U(-I)-C-C
        # U-bend: (I-)C-C
        super().__init__()

        max_depth = min(max_depth, len(topologies))
        self.topologies = topologies[-max_depth:]

        unet_channels = (
            make_unet_channels(channels, max_depth)
            if isinstance(channels, int)
            else channels
        )
        assert isinstance(unet_channels, dict)

        in_ch = in_channels

        # Encoder
        self.encoder = torch.nn.ModuleList()
        pool_topologies = self.topologies[::-1]
        skip_channels = []
        for i, (out_ch, topo) in enumerate(
            zip(unet_channels["encoder"], pool_topologies)
        ):
            do_pool = i < max_depth - 1
            self.encoder.append(
                EncoderUnit(
                    conv_kwargs=dict(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        conv_module=conv_module,
                        topology=topo,
                        n=n_conv,
                    ),
                    pool_kwargs=dict(topology=topo, reduce=reduce),
                    do_pool=do_pool,
                )
            )
            if do_pool:
                skip_channels.append(out_ch)
            in_ch = out_ch
        skip_channels = skip_channels[::-1]

        # Decoder
        unpool_topologies = self.topologies[:-1]
        conv_topologies = self.topologies[1:]
        self.decoder = torch.nn.ModuleList()
        for out_ch, skip_ch, top_unpool, top_conv in zip(
            unet_channels["decoder"], skip_channels, unpool_topologies, conv_topologies
        ):
            self.decoder.append(
                DecoderUnit(
                    conv_kwargs=dict(
                        in_channels=in_ch + skip_ch,
                        out_channels=out_ch,
                        conv_module=conv_module,
                        topology=top_conv,
                        n=n_conv,
                    ),
                    unpool_kwargs=dict(topology=top_unpool, reduce=reduce),
                )
            )
            in_ch = out_ch
        self.out_ch = out_ch

    def get_prediction_topology(self):
        return self.topologies[-1]

    def forward(self, features):
        # Encoder
        skip_features = []
        for enc_unit in self.encoder:
            features, sf = enc_unit(features)
            if enc_unit.do_pool:
                skip_features.append(sf)

        skip_features = skip_features[::-1]

        # Decoder
        for dec_unit, sf in zip(self.decoder, skip_features):
            features = dec_unit(features, sf)

        return features
