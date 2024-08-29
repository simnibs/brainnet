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
        spatial_dims,
        in_channels,
        encoder_channels,
        decoder_channels,
        max_pool_size = 2,
        # deep_features = None, # [[False], [False], [False]], [[False, True]]
        # channels=64,
        # levels=None,
        # n_convs=1,
        # multiplier=1,
    ):

        super().__init__()

        self.num_levels = len(encoder_channels)

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
        for i, level in enumerate(encoder_channels):
            conv_block = torch.nn.ModuleList()
            for out_ch in level:
                conv_block.append(ConvBlock(spatial_dims, in_ch, out_ch))
                in_ch = out_ch
            self.encoder.append(conv_block)
            if add_skip_connection(i):
                skip_connections.append(in_ch)

        # Decoder (upsampling path)
        self.decoder = torch.nn.ModuleList()
        for i, level in enumerate(decoder_channels):
            in_ch += skip_connections.pop()
            conv_block = torch.nn.ModuleList()
            for out_ch in level:
                conv_block.append(ConvBlock(spatial_dims, in_ch, out_ch))
                in_ch = out_ch
            self.decoder.append(conv_block)

        self.final_channels = out_ch

    def forward(self, features):
        do_pooling = lambda level: level < (self.num_levels - 1)

        # Encoder
        skip_connections = []
        for i, conv_blocks in enumerate(self.encoder):
            for block in conv_blocks:
                features = block(features)
            if do_pooling(i):
                skip_connections.append(features)
                features = self.pooling(features)

        # Decoder
        for i, conv_blocks in enumerate(self.decoder):
            features = self.upsampling(features)
            features = torch.cat([features, skip_connections.pop()], dim=1)
            for block in conv_blocks:
                features = block(features)

        return features
