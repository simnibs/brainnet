import torch

class ConvolutionBlock(torch.nn.Module):
    """
    Specific convolutional block followed by PReLU for unet.
    """

    def __init__(
            self,
            ndim,
            in_channels,
            out_channels,
            norm: bool = True,
            activation: bool = True,
            p_dropout=0.0,
            init_zeros: bool = False
        ):
        super().__init__()
        stride = 1
        kernel_size = 3
        assert kernel_size % 2 == 1
        padding = int((kernel_size - 1) / 2)

        # Fixed order of normalization, activation, and dropout
        # conv -> norm -> activation -> drop out
        Conv = getattr(torch.nn, f"Conv{ndim}d")
        convolution = Conv(in_channels, out_channels, kernel_size, stride, padding)
        if init_zeros:
            torch.nn.init.zeros_(convolution.weight)
            torch.nn.init.zeros_(convolution.bias)

        self.transform = torch.nn.Sequential()
        self.transform.append(convolution)
        if norm:
            self.transform.append(getattr(torch.nn, f"InstanceNorm{ndim}d")(out_channels))
        if activation:
            self.transform.append(torch.nn.PReLU())
        if p_dropout > 0.0:
            self.transform.append(getattr(torch.nn, f"Dropout{ndim}d"))(p_dropout)

    def forward(self, x):
        return self.transform(x)

# class ConvBlock(torch.nn.Sequential):
#     """
#     Specific convolutional block followed by PReLU for unet.
#     """

#     def __init__(
#             self,
#             ndim,
#             in_channels,
#             out_channels,
#             norm: bool = True,
#             activation: bool = True,
#             dropout_prob=0.0,
#             init_zeros: bool = False
#         ):
#         super().__init__()
#         stride = 1
#         kernel_size = 3
#         assert kernel_size % 2 == 1
#         padding = int((kernel_size - 1) / 2)

#         # conv -> norm -> activation -> drop out
#         Conv = getattr(torch.nn, f"Conv{ndim}d")
#         convolution = Conv(in_channels, out_channels, kernel_size, stride, padding)
#         if init_zeros:
#             torch.nn.init.zeros_(convolution.weight)
#             torch.nn.init.zeros_(convolution.bias)

#         self.append(convolution)
#         if norm:
#             self.append(getattr(torch.nn, f"InstanceNorm{ndim}d")(out_channels))
#         if activation:
#             self.append(torch.nn.PReLU())
#         if dropout_prob > 0.0:
#             self.append(getattr(torch.nn, f"Dropout{ndim}d"))(dropout_prob)

