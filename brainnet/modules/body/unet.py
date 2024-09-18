import torch

from brainnet.modules.blocks import ConvBlock


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
        max_pool_size: int = 2,
        return_encoder_features: (
            None | list[bool]
        ) = None,  # [False, False, False, False, True]
        return_decoder_features: None | list[bool] = None,  # [True, True, True, True]
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

        self.num_levels = len(encoder_channels)

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

        add_skip_connection = lambda level: level < (self.num_levels - 1)

        # Downsampling (Pooling)
        MaxPooling = getattr(torch.nn, "MaxPool%dd" % spatial_dims)
        self.pooling = MaxPooling(max_pool_size)

        # Upsampling
        self.upsampling = torch.nn.Upsample(scale_factor=max_pool_size, mode="nearest")

        # Encoder (downsampling path)
        in_ch = in_channels
        skip_connections = []
        self.encoder = torch.nn.ModuleList()
        self.encoder_scale = []
        for i, level in enumerate(encoder_channels):
            conv_block = torch.nn.ModuleList()
            for out_ch in level:
                conv_block.append(ConvBlock(spatial_dims, in_ch, out_ch))
                in_ch = out_ch
            self.encoder.append(conv_block)
            if add_skip_connection(i):
                skip_connections.append(in_ch)
            self.encoder_scale.append(int(max_pool_size**i))

        # Decoder (upsampling path)
        self.decoder = torch.nn.ModuleList()
        self.decoder_scale = []
        scale = self.encoder_scale[-1]
        for i, level in enumerate(decoder_channels):
            in_ch += skip_connections.pop()
            conv_block = torch.nn.ModuleList()
            for out_ch in level:
                conv_block.append(ConvBlock(spatial_dims, in_ch, out_ch))
                in_ch = out_ch
            self.decoder.append(conv_block)
            scale /= max_pool_size
            self.decoder_scale.append(int(scale))

        self.final_channels = out_ch

        self.do_pooling = lambda level: level < (self.num_levels - 1)

        self.feature_scales = [
            s
            for i, s in zip(
                self.return_encoder_features + self.return_decoder_features,
                self.encoder_scale + self.decoder_scale,
            )
            if i
        ]

    def upsample_feature(self, feature, scale):
        return torch.nn.functional.interpolate(
            feature,
            scale_factor=scale,
            mode="trilinear",
            align_corners=True,
        )

    def sum_features(self, features):
        """Upsample and concatenate all features.

        NOTE This can take up a *lot* of memory depending on the spatial size
        of the output layer!

        """
        if len(features) == 1:
            return features[0]
        else:
            out = torch.zeros_like(features[-1])
            for f,s in zip(features, self.feature_scales):
                out = out + self.upsample_feature(f, s)
            return out

    def cat_features(self, features):
        if len(features) == 1:
            return features[0]
        else:
            return torch.cat(
                [
                    self.upsample_feature(f, s)
                    for f, s in zip(features, self.feature_scales)
                ],
                dim=1,
            )

    def forward(self, features):

        unet_features = []

        # Encoder
        skip_connections = []
        for i, conv_blocks in enumerate(self.encoder):
            for block in conv_blocks:
                features = block(features)
            if self.return_encoder_features[i]:
                unet_features.append(features)
            if self.do_pooling(i):
                skip_connections.append(features)
                features = self.pooling(features)

        # Decoder
        for i, conv_blocks in enumerate(self.decoder):
            features = self.upsampling(features)
            features = torch.cat([features, skip_connections.pop()], dim=1)
            for block in conv_blocks:
                features = block(features)
            # the last features are returned anyway
            if self.return_decoder_features[i] and not i == (self.num_levels - 1):
                unet_features.append(features)

        return unet_features
