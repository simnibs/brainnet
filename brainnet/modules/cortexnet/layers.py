import torch

class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

class GraphConv(torch.nn.Module):
    """Graph convolution block from Morris et al. (2018) as presented in eq. 5.

    Parameters
    ----------


    References
    ----------
    Morris et al. (2018). Weisfeiler and Leman Go Neural: Higher-order
        Graph Neural Networks. https://arxiv.org/abs/1810.02244.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduce_index: torch.Tensor,
        gather_index: torch.Tensor,
        bias = False,
        init_zeros: bool = False,
        norm: None | torch.nn.Module = torch.nn.BatchNorm1d,
        activation: None | torch.nn.Module = torch.nn.PReLU,
        norm_kwargs = None,
        activation_kwargs = None,
        # dropout_p=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduce_index = reduce_index.long()  # repeats of vertex
        self.gather_index = gather_index.long()  # neighbors of vertex

        kwargs = dict(kernel_size=1, stride=1, padding="valid", bias=bias)
        self.conv_v = torch.nn.Conv1d(in_channels, out_channels, **kwargs)
        self.conv_n = torch.nn.Conv1d(in_channels, out_channels, **kwargs)

        if init_zeros:
            torch.nn.init.zeros_(self.conv_v.weight)
            torch.nn.init.zeros_(self.conv_n.weight)
            if bias:
                torch.nn.init.zeros_(self.conv_v.bias)
                torch.nn.init.zeros_(self.conv_n.bias)

        norm_kwargs = norm_kwargs or {}
        self.norm = IdentityModule() if norm is None else norm(out_channels, **norm_kwargs)

        activation_kwargs = activation_kwargs or {}
        self.activation =  IdentityModule() if activation is None else activation(**activation_kwargs)

        # self.apply_dropout = dropout_p > 0.0
        # if self.apply_dropout:
        #     self.dropout = torch.nn.Dropout1d(dropout_p)

    def forward(self, in_features):
        # F_in : (batch, channels, vertices)

        # F_v = self.linear0(in_features)  # feature for vertex as center
        # F_n = self.linear1(in_features)  # feature for vertex as neighbor

        F_v = self.conv_v(in_features)  # feature for vertex as center
        F_n = self.conv_n(in_features)  # feature for vertex as neighbor

        # aggregate (neighbors): mean
        out = torch.zeros_like(F_v)
        out.index_reduce_(
            dim=2,
            index=self.reduce_index,
            source=F_n[..., self.gather_index],
            reduce="mean",
            include_self=False,
        )
        # out.index_add_(
        #     dim=2,
        #     index=self.reduce_index,
        #     source=F_n[..., self.gather_index],
        # )
        # merge (v and neighbors): sum
        out = out + F_v
        out = self.norm(out)
        out = self.activation(out)
        return out


class ResidualGraphConv(torch.nn.Module):
    def __init__(
        self,
        channels: list[int] | tuple, # [64, 64, 64, 64]
        reduce_index: torch.Tensor,
        gather_index: torch.Tensor,
        bias=False,
    ) -> None:
        super().__init__()
        assert channels[0] == channels[-1], "Last channel does not match residual"

        self.convs = torch.nn.Sequential()
        for in_ch, out_ch in zip(channels[:-2], channels[1:-1]):
            self.convs.append(GraphConv(in_ch, out_ch, reduce_index, gather_index, bias))

        self.last_conv = GraphConv(channels[-2], channels[-1], reduce_index, gather_index, bias, activation=None)
        self.last_activation = torch.nn.PReLU()

    def forward(self, features):
        residual = features
        for conv in self.convs:
            features = conv(features)
        features = self.last_activation(self.last_conv(features) + residual)
        return features


class GraphDeformationBlock(torch.nn.Module):
    def __init__(self,
        image_channels: int, # 256
        graph_channels: list[int] | tuple, # [64, 64, 64, 64]
        out_channels: int,
        reduce_index,
        gather_index,
        n_residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index)
        self.convs = torch.nn.Sequential(
            # image features -> graph features
            GraphConv(image_channels, graph_channels[0], **kw),
            # features -> features
            *[ResidualGraphConv(graph_channels, **kw) for _ in range(n_residual_blocks)],
            # features -> deformation field
            GraphConv(graph_channels[-1], out_channels, **kw, init_zeros=True),
        )

    def forward(self, features):
        return self.convs(features)



