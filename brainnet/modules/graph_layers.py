import torch


class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x


class GraphConvolution(torch.nn.Module):
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
        bias=True,
        init_zeros: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduce_index = reduce_index.long()  # repeats of vertex
        self.gather_index = gather_index.long()  # neighbors of vertex

        kwargs = dict(kernel_size=1, stride=1, bias=bias)

        # Compute features when a vertex is the position of interest
        self.conv_self = torch.nn.Conv1d(in_channels, out_channels, **kwargs)
        # Compute features when a vertex is part of the neighborhood
        self.conv_other = torch.nn.Conv1d(in_channels, out_channels, **kwargs)

        if init_zeros:
            torch.nn.init.zeros_(self.conv_self.weight)
            torch.nn.init.zeros_(self.conv_other.weight)
            if bias:
                torch.nn.init.zeros_(self.conv_self.bias)
                torch.nn.init.zeros_(self.conv_other.bias)
        # else:
        #     std = 1e-5
        #     torch.nn.init.normal_(self.conv_v.weight, std=std)
        #     torch.nn.init.normal_(self.conv_n.weight, std=std)
        #     if bias:
        #         torch.nn.init.normal_(self.conv_v.bias, std=std)
        #         torch.nn.init.normal_(self.conv_n.bias, std=std)

    def forward(self, features):

        features_self = self.conv_self(features)    # W0
        features_other = self.conv_other(features)  # W1

        # aggregate (neighbors)
        out = torch.zeros_like(features_self)
        # out.index_reduce_(
        #     dim=2,
        #     index=self.reduce_index,
        #     source=F_n[..., self.gather_index],
        #     reduce="mean",
        #     include_self=False,
        # )
        out.index_add_(
            dim=-1,
            index=self.reduce_index,
            source=features_other[..., self.gather_index],
        )
        # merge (self and neighbors)
        return out + features_self

class EdgeConvolution(GraphConvolution):

    def forward(self, features):

        features_self = self.conv_self(features)    # W0
        features_other = self.conv_other(features)  # W1

        out = torch.zeros_like(features_other)
        out.index_reduce_(
            dim=-1,
            index=self.reduce_index,
            source=features_other[..., self.gather_index],
            reduce="mean",
            include_self=False,
        )
        return features_self - features_other + out


# class EdgeConv(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         reduce_index: torch.Tensor,
#         gather_index: torch.Tensor,
#         # batchnorm=True,
#         bias: bool = True,
#         init_zeros: bool = False,
#     ):
#         super().__init__()
#         self.in_channels = 2 * in_channels  # vertex and neighbor concatenation
#         self.out_channels = out_channels
#         self.reduce_index = reduce_index.long()  # repeats of vertex
#         self.gather_index = gather_index.long()  # neighbors of vertex

#         # self.linear = torch.nn.Linear(in_channels, out_channels, bias)
#         self.conv = torch.nn.Conv1d(self.in_channels, out_channels, 1, bias=bias)

#         if init_zeros:
#             torch.nn.init.zeros_(self.conv.weight)
#             if bias:
#                 torch.nn.init.zeros_(self.conv.bias)

#         self.activation = torch.nn.PReLU()

#         # self.batchnorm = torch.nn.BatchNorm1d(out_channels) if batchnorm else None

#         # self.apply_dropout = dropout_p > 0.0
#         # if self.apply_dropout:
#         #     self.dropout = torch.nn.Dropout1d(dropout_p)

#     def forward(self, in_features):
#         # F_in : (batch, channels, vertices)
#         batch_size, _, n_vertices = in_features.shape
#         out_shape = batch_size, self.out_channels, n_vertices

#         vertices = in_features[..., self.reduce_index]
#         neighbors = in_features[..., self.gather_index]

#         concat_features = torch.cat([vertices, neighbors - vertices], dim=1)
#         F_e = self.conv(concat_features)



#         # Index pooling of features
#         out = torch.zeros(out_shape, dtype=F_e.dtype, device=in_features.device)
#         out.index_reduce_(
#             dim=-1,
#             index=self.reduce_index,
#             source=F_e,
#             reduce="mean",
#             include_self=False,
#         )

#         return self.activation(out)


def convolution_block(Convolution):
    class ConvolutionBlock(Convolution):
        """Graph convolution followed by normalization, activation, and dropout."""
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reduce_index: torch.Tensor,
            gather_index: torch.Tensor,
            bias: bool = True,
            init_zeros: bool = False,
            norm: None | type[torch.nn.Module] = torch.nn.InstanceNorm1d,
            activation: None | type[torch.nn.Module] = torch.nn.PReLU,
            norm_kwargs: dict | None = None,
            activation_kwargs: dict | None = None,
            dropout_prob: float = 0.0,
        ):
            super().__init__(
                in_channels,
                out_channels,
                reduce_index,
                gather_index,
                bias,
                init_zeros,
            )
            # norm -> activation -> drop out
            self.NAD = torch.nn.Sequential()
            if norm is not None:
                norm_kwargs = norm_kwargs or {}
                if norm == torch.nn.InstanceNorm1d:
                    shape = out_channels
                    norm_kwargs |= dict(affine=False, track_running_stats=False)
                elif norm == torch.nn.LayerNorm:
                    shape = (out_channels, int(reduce_index.amax() + 1))
                    norm_kwargs |= dict(elementwise_affine=False)
                else:
                    raise ValueError(f"{norm} is currently not supported.")
                self.NAD.append(norm(shape, **norm_kwargs))
            if activation is not None:
                activation_kwargs = activation_kwargs or {}
                self.NAD.append(activation(**activation_kwargs))
            if dropout_prob > 0.0:
                self.NAD.append(torch.nn.Dropout1d(dropout_prob))

        def forward(self, features):
            return self.NAD(super().forward(features))

    return ConvolutionBlock

GraphConvolutionBlock = convolution_block(GraphConvolution)
EdgeConvolutionBlock = convolution_block(EdgeConvolution)

class ResidualGraphConv(torch.nn.Module):
    def __init__(
        self,
        channels: list[int] | tuple,  # [64, 64, 64, 64]
        reduce_index: torch.Tensor,
        gather_index: torch.Tensor,
    ) -> None:
        super().__init__()
        assert channels[0] == channels[-1], "Last channel does not match residual"

        kwargs = dict(reduce_index=reduce_index, gather_index=gather_index)

        self.convs = torch.nn.Sequential()
        for in_ch, out_ch in zip(channels[:-2], channels[1:-1]):
            self.convs.append(GraphConvolutionBlock(in_ch, out_ch, **kwargs))
        self.last_conv = GraphConvolutionBlock(
            channels[-2], channels[-1], activation=None, **kwargs
        )
        self.norm = torch.nn.InstanceNorm1d(channels[0], affine=False, track_running_stats=False)
        self.last_activation = torch.nn.PReLU()

    def forward(self, features):
        residual = features
        for conv in self.convs:
            features = conv(features)
        features = self.last_activation(self.last_conv(features) + self.norm(residual))
        return features


class GraphConvolutionDeformationBlock(torch.nn.Module):
    def __init__(
        self,
        image_channels: int,  # 256
        graph_channels: list[int] | tuple,  # [64, 64, 64]
        out_channels: int,
        reduce_index,
        gather_index,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index)
        self.convs = torch.nn.Sequential(
            # image features -> graph features
            GraphConvolutionBlock(image_channels, graph_channels[0], **kw),
            # features -> features
            *[
                GraphConvolutionBlock(in_ch, out_ch, **kw)
                for in_ch, out_ch in zip(graph_channels[:-1], graph_channels[1:])
            ],
            # features -> deformation field
            GraphConvolution(graph_channels[-1], out_channels, **kw, init_zeros=True),
        )

    def forward(self, features):
        return self.convs(features)


class ResidualGraphConvolutionDeformationBlock(torch.nn.Module):
    def __init__(
        self,
        image_channels: int,  # 256
        graph_channels: list[int] | tuple,  # [64, 64, 64, 64]
        out_channels: int,
        reduce_index,
        gather_index,
        n_residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index)
        self.convs = torch.nn.Sequential(
            # image features -> graph features
            GraphConvolutionBlock(image_channels, graph_channels[0], **kw),
            # features -> features
            *[
                ResidualGraphConv(graph_channels, **kw)
                for _ in range(n_residual_blocks)
            ],
            # features -> deformation field
            GraphConvolution(graph_channels[-1], out_channels, **kw, init_zeros=True),
        )

    def forward(self, features):
        return self.convs(features)

class nConv(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_module: torch.nn.Module,
        topology,
        n: int = 1,
    ) -> None:
        super().__init__()
        assert n > 0
        reduce_index, gather_index = topology.get_convolution_indices()

        for _ in torch.arange(n):
            self.append(
                conv_module(in_channels, out_channels, reduce_index, gather_index),
            )
            in_channels = out_channels


# class BlockRepeater(torch.nn.Sequential):
#     def __init__(self, repeats: int, module, *args, **kwargs):
#         assert repeats > 0
#         for _ in torch.arange(n):
#             self.append(
#                 module(in_channels, out_channels, reduce_index, gather_index),
#             )
#             in_channels = out_channels


class GraphPool(torch.nn.Module):
    def __init__(self, topology, reduce: str) -> None:
        super().__init__()
        self.topology = topology
        self.reduce = reduce

    def forward(self, features):
        return self.topology.pool(features, self.reduce)


class GraphUnpool(torch.nn.Module):
    def __init__(self, topology, reduce: str) -> None:
        super().__init__()
        self.topology = topology
        self.reduce = reduce

    def forward(self, features):
        return self.topology.unpool(features, self.reduce)


# class GraphEncoderUnit(torch.nn.Sequential):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         topology: Topology,
#         conv_module: torch.nn.Module,
#         reduce: str = "amax",
#         n_conv: int = 1,
#         pool: bool = True,
#     ) -> None:
#         """A single"""
#         super().__init__()
#         self.add_module(
#             "nconv",
#             nConv(in_channels, out_channels, conv_module, topology, n_conv),
#         )
#         self.add_module("pool", GraphPool(topology, reduce))


# class GraphDecoderUnit(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         topology: Topology,
#         conv_module: torch.nn.Module,
#         reduce: str = "amax",
#         n_conv: int = 1,
#         unpool: bool = True,
#     ) -> None:
#         """
#         conv_module : torch.nn.Module
#             E.g., GraphConv or EdgeConv
#         """
#         super().__init__()
#         self.nconv = nConv(in_channels, out_channels, conv_module, topology, n_conv)
#         self.unpool = GraphUnpool(topology, reduce) if unpool else lambda x: x

#     def forward(self, a: torch.Tensor, b: Union[None,torch.Tensor] = None):
#         """Concatenate `a` with `b`, feed through convolutional layer(s), and
#         unpool.
#         """
#         if b is None:
#             a = torch.cat((a, b), dim=1)
#         out = self.nconv(a)
#         out = self.unpool(out)
#         return out

