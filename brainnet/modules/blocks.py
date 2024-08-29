import torch

class ConvBlock(torch.nn.Module):
    """
    Specific convolutional block followed by PReLU for unet.
    """

    def __init__(self, ndim, in_channels, out_channels, stride=1, dropout_p=0.0, init_zeros: bool = False):
        super().__init__()

        self.apply_dropout = dropout_p > 0.0

        Conv = getattr(torch.nn, f"Conv{ndim}d")
        self.convolution = Conv(in_channels, out_channels, 3, stride, 1)
        if init_zeros:
            torch.nn.init.zeros_(self.convolution.weight)
            torch.nn.init.zeros_(self.convolution.bias)

        # self.activation = torch.nn.LeakyReLU(0.2)
        self.activation = torch.nn.PReLU()
        if self.apply_dropout:
            dropout = getattr(torch.nn, f"Dropout{ndim}d")
            self.dropout = dropout(dropout_p)
        # self.norm = torch.nn.InstanceNorm3d()

    def forward(self, x):
        out = self.convolution(x)
        out = self.activation(out)
        out = self.dropout(out) if self.apply_dropout else out
        # out = self.norm(out)
        return out
