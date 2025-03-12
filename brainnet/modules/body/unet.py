import torch

from brainnet.modules.blocks import ConvolutionBlock


def make_unet_path(channels, levels, convs=1, multiplier=2):
    if levels == 1:
        return [[channels for _ in range(convs)]]
    return [[channels for _ in range(convs)]] + make_unet_path(
        int(channels * multiplier), levels - 1, convs, multiplier
    )


def make_unet_paths(channels: int, levels, convs=1, multiplier=2):
    """

    channels :
        Initial (and final) number of channels.
    levels :
        Number of levels.
    convs :
        Number of convolutions per level.
    multipler :
        Multiplier at each level.

    Returns
    -------

    """
    encoder_channels = make_unet_path(channels, levels, convs, multiplier)
    decoder_channels = make_unet_path(
        encoder_channels[-2][-1], levels - 1, convs, 1 / multiplier
    )

    return encoder_channels, decoder_channels


class UNet(torch.nn.Module):
    """
    Image U-Net architecture.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        encoder_channels: list[list[int]],
        decoder_channels: list[list[int]],
        return_encoder_features: (None | list[bool]) = None,
        return_decoder_features: None | list[bool] = None,
        max_pool_size: int = 2,
        encoder_post: list[list[int]] | None = None,
        decoder_post: list[list[int]] | None = None,
    ):
        """_summary_

        Parameters
        ----------
        spatial_dims : _type_
            _description_
        in_channels : _type_
            _description_
        encoder_channels : _type_
            _description_
        decoder_channels : _type_
            _description_
        max_pool_size : int
            _description_, by default 2
        return_encoder_features : None | list[bool]
            The features from the final layer are *always* returned.
        """
        super().__init__()
        d = spatial_dims
        self.num_levels = len(encoder_channels)

        if encoder_post is None:
            encoder_post = [[None]] * self.num_levels
        if decoder_post is None:
            decoder_post = [[None]] * (self.num_levels - 1)
        assert (
            isinstance(encoder_post, (list, tuple))
            and len(encoder_post) == self.num_levels
        )
        assert (
            isinstance(decoder_post, (list, tuple))
            and len(decoder_post) == self.num_levels - 1
        )

        if return_encoder_features is not None:
            self.return_encoder_features = return_encoder_features
        else:
            self.return_encoder_features = self.num_levels * [False]

        if return_decoder_features is not None:
            self.return_decoder_features = return_decoder_features
        else:
            self.return_decoder_features = (self.num_levels - 1) * [False]

        # we always return the final features!
        self.return_decoder_features[-1] = True

        # Downsampling (Pooling)
        MaxPooling = getattr(torch.nn, "MaxPool%dd" % spatial_dims)
        self.pooling = MaxPooling(max_pool_size)

        # Upsampling
        self.upsampling = torch.nn.Upsample(scale_factor=max_pool_size, mode="nearest")

        self.num_features = {}

        # Encoder (downsampling path)
        in_ch = in_channels
        skip_connections = []

        self.encoder = torch.nn.ModuleDict()
        self.encoder_post = torch.nn.ModuleDict()
        self.encoder_scale = []
        for i, e_chs in enumerate(encoder_channels):
            self.encoder[f"enc:{i}"], out_ch = self.make_conv_block(d, in_ch, e_chs)
            # Add skip connection
            if i < (self.num_levels - 1):
                skip_connections.append(out_ch)
            self.encoder_scale.append(int(max_pool_size**i))
            in_ch = out_ch
            # Post processing
            if self.return_encoder_features[i]:
                self.encoder_post[f"enc:{i}"], out_ch = self.make_conv_block(
                    d, out_ch, encoder_post[i]
                )
            self.num_features[f"enc:{i}"] = out_ch

        # Decoder (upsampling path)
        self.decoder = torch.nn.ModuleDict()
        self.decoder_post = torch.nn.ModuleDict()
        self.decoder_scale = []
        scale = self.encoder_scale[-1]
        for i, d_chs in enumerate(decoder_channels):
            in_ch += skip_connections.pop()
            self.decoder[f"dec:{i}"], out_ch = self.make_conv_block(d, in_ch, d_chs)
            scale /= max_pool_size
            self.decoder_scale.append(int(scale))
            in_ch = out_ch
            # Post processing
            if self.return_decoder_features[i]:
                self.decoder_post[f"dec:{i}"], out_ch = self.make_conv_block(
                    d, out_ch, decoder_post[i]
                )
            self.num_features[f"dec:{i}"] = out_ch

        self.final_channels = out_ch

        self.feature_scales = [
            s
            for i, s in zip(
                self.return_encoder_features + self.return_decoder_features,
                self.encoder_scale + self.decoder_scale,
            )
            if i
        ]

        self.encoder_features = [
            f"enc:{i}" for i, b in enumerate(self.return_encoder_features) if b
        ]
        self.decoder_features = [
            f"dec:{i}" for i, b in enumerate(self.return_decoder_features) if b
        ]

        # self.feature_scales = {f"{self._fm_name['encoder'](i)}": n for i,n in enumerate(self.encoder_scale)}
        # self.feature_scales |= {f"{self._fm_name['decoder'](i)}": n for i,n in enumerate(self.decoder_scale)}

    @staticmethod
    def make_conv_block(spatial_dims, in_ch, channels: list[int] | tuple | None = None):
        conv_block = torch.nn.Sequential()
        if channels[0] is not None:
            for out_ch in channels:
                conv_block.append(ConvolutionBlock(spatial_dims, in_ch, out_ch))
                in_ch = out_ch
        return conv_block, in_ch

    def upsample_feature(self, feature, scale):
        return torch.nn.functional.interpolate(
            feature,
            scale_factor=scale,
            mode="trilinear",
            align_corners=True,
        )

    # def sum_features(self, features: dict[str, torch.Tensor]):

    #     if len(features) == 1:
    #         return features["decoder:3"]
    #     else:
    #         out = torch.zeros_like(features[-1])
    #         for f,s in zip(features, self.feature_scales):
    #             out = out + self.upsample_feature(f, s)
    #         return out

    # def cat_features(self, features):
    #     """Upsample and concatenate all features.

    #     NOTE This can take up a *lot* of memory depending on the spatial size
    #     of the output layer!

    #     """
    #     if len(features) == 1:
    #         return features[0]
    #     else:
    #         return torch.cat(
    #             [
    #                 self.upsample_feature(f, s)
    #                 for f, s in zip(features, self.feature_scales)
    #             ],
    #             dim=1,
    #         )

    def forward(self, features):
        unet_features = dict()

        # Encoder
        skip_connections = []
        for i, (name, conv_blocks) in enumerate(self.encoder.items()):
            features = conv_blocks(features)
            if self.return_encoder_features[i]:
                unet_features[name] = self.encoder_post[name](features)
            # Pool
            if i < (self.num_levels - 1):
                skip_connections.append(features)
                features = self.pooling(features)

        # Decoder
        for i, (name, conv_blocks) in enumerate(self.decoder.items()):
            features = self.upsampling(features)
            features = torch.cat([features, skip_connections.pop()], dim=1)
            features = conv_blocks(features)
            # the last features are returned anyway
            if self.return_decoder_features[i] and not i == (self.num_levels - 1):
                unet_features[name] = self.decoder_post[name](features)

        return unet_features
