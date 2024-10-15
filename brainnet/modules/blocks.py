import torch

class ConvBlock(torch.nn.Module):
    """
    Specific convolutional block followed by PReLU for unet.
    """

    def __init__(self, ndim, in_channels, out_channels, dropout_prob=0.0, init_zeros: bool = False):
        super().__init__()
        stride = 1
        kernel_size = 3
        assert kernel_size % 2 == 1
        padding = int((kernel_size - 1) / 2)

        # conv -> norm -> activation -> drop out
        Conv = getattr(torch.nn, f"Conv{ndim}d")
        convolution = Conv(in_channels, out_channels, kernel_size, stride, padding)
        if init_zeros:
            torch.nn.init.zeros_(convolution.weight)
            torch.nn.init.zeros_(convolution.bias)
        self.transform = torch.nn.Sequential(
            convolution,
            getattr(torch.nn, f"InstanceNorm{ndim}d")(out_channels),
            torch.nn.PReLU()
        )
        if dropout_prob > 0.0:
            self.transform.append(getattr(torch.nn, f"Dropout{ndim}d"))(dropout_prob)

    def forward(self, x):
        return self.transform(x)

# class ConvBlock(torch.nn.Module):
#     """
#     Specific convolutional block followed by PReLU for unet.
#     """

#     def __init__(self, ndim, in_channels, out_channels, dropout_prob=0.0, init_zeros: bool = False):
#         super().__init__()
#         stride = 1
#         kernel_size = 3
#         assert kernel_size % 2 == 1
#         padding = int((kernel_size - 1) / 2)

#         Conv = getattr(torch.nn, f"Conv{ndim}d")
#         self.convolution = Conv(in_channels, out_channels, kernel_size, stride, padding)
#         if init_zeros:
#             torch.nn.init.zeros_(self.convolution.weight)
#             torch.nn.init.zeros_(self.convolution.bias)

#         self.activation = torch.nn.PReLU()

#     def forward(self, x):
#         out = self.convolution(x)
#         out = self.activation(out)
#         # out = self.dropout(out) if self.apply_dropout else out
#         return out
