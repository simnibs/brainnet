import torch


class ConvolutionBlock(torch.nn.Sequential):
    """
    Convolution, normalization, activation, dropout block.
    """

    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        norm: bool = True,
        activation: bool = True,
        dropout_prob=0.0,
        init_zeros: bool = False,
        conv_kwargs: dict | None = None,
    ):
        super().__init__()
        stride = 1
        kernel_size = 3
        default_conv_kwargs = dict(
            stride=stride, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2)
        )
        if conv_kwargs is not None:
            default_conv_kwargs |= conv_kwargs

        # conv -> norm -> activation -> drop out
        Conv = getattr(torch.nn, f"Conv{ndim}d")
        convolution = Conv(in_channels, out_channels, **default_conv_kwargs)
        if init_zeros:
            torch.nn.init.zeros_(convolution.weight)
            torch.nn.init.zeros_(convolution.bias)

        self.append(convolution)
        if norm:
            self.append(getattr(torch.nn, f"InstanceNorm{ndim}d")(out_channels))
        if activation:
            self.append(torch.nn.PReLU())
        if dropout_prob > 0.0:
            self.append(getattr(torch.nn, f"Dropout{ndim}d"))(dropout_prob)


class TransposedConvolutionBlock(torch.nn.Sequential):
    """
    Transposed convolution, normalization, activation, dropout block.
    """

    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        norm: bool = True,
        activation: bool = True,
        dropout_prob=0.0,
        init_zeros: bool = False,
        conv_kwargs: dict | None = None,
    ):
        super().__init__()
        kernel_size = 3
        default_conv_kwargs = dict(
            stride=1,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            output_padding=1,
        )
        if conv_kwargs is not None:
            default_conv_kwargs |= conv_kwargs

        # conv -> norm -> activation -> drop out
        Conv = getattr(torch.nn, f"TransposedConv{ndim}d")
        convolution = Conv(in_channels, out_channels, **default_conv_kwargs)
        if init_zeros:
            torch.nn.init.zeros_(convolution.weight)
            torch.nn.init.zeros_(convolution.bias)

        self.append(convolution)
        if norm:
            self.append(getattr(torch.nn, f"InstanceNorm{ndim}d")(out_channels))
        if activation:
            self.append(torch.nn.PReLU())
        if dropout_prob > 0.0:
            self.append(getattr(torch.nn, f"Dropout{ndim}d"))(dropout_prob)
