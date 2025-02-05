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
        features_self = self.conv_self(features)  # W0
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
        features_self = self.conv_self(features)  # W0
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


class ResidualGraphConvolution(torch.nn.Module):
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
        self.norm = torch.nn.InstanceNorm1d(
            channels[0], affine=False, track_running_stats=False
        )
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
        in_channels: int,
        graph_channels: list[int],
        out_channels: int,
        reduce_index,
        gather_index,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index)
        # input features -> features
        # features -> features
        for i, (in_ch, out_ch) in enumerate(
            zip([in_channels] + graph_channels[:-1], graph_channels)
        ):
            self.add_module(
                f"GraphConvolution:{i}", GraphConvolutionBlock(in_ch, out_ch, **kw)
            )
        # features -> deformation field
        self.add_module(
            f"GraphConvolution:{i + 1}",
            GraphConvolution(graph_channels[-1], out_channels, **kw, init_zeros=True),
        )


class EdgeConvolutionDeformationBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        graph_channels: list[int],
        out_channels: int,
        reduce_index,
        gather_index,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index)
        # input features -> features
        # features -> features
        for i, (in_ch, out_ch) in enumerate(
            zip([in_channels] + graph_channels[:-1], graph_channels)
        ):
            self.add_module(
                f"GraphConvolution:{i}", EdgeConvolutionBlock(in_ch, out_ch, **kw)
            )
        # features -> deformation field
        self.add_module(
            f"GraphConvolution:{i + 1}",
            EdgeConvolution(graph_channels[-1], out_channels, **kw, init_zeros=True),
        )


class ResidualGraphConvolutionDeformationBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        graph_channels: list[int],
        out_channels: int,
        reduce_index,
        gather_index,
        n_residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index)
        # input features -> features
        self.add_module(
            "GraphConvolution:0", GraphConvolution(in_channels, graph_channels[0], **kw)
        )
        # features -> features
        for i in range(n_residual_blocks):
            self.add_module(
                f"ResidualGraphConvolution:{i}",
                ResidualGraphConvolution(graph_channels, **kw),
            )
        # features -> deformation field
        self.add_module(
            "GraphConvolution:1",
            GraphConvolution(graph_channels[-1], out_channels, **kw, init_zeros=True),
        )


class Pool(torch.nn.Module):
    def __init__(self, topology, reduce: str) -> None:
        super().__init__()
        self.topology = topology
        self.reduce = reduce

    def forward(self, features):
        return self.topology.pool(features, self.reduce)


class Unpool(torch.nn.Module):
    def __init__(self, topology, reduce: str) -> None:
        super().__init__()
        self.topology = topology
        self.reduce = reduce

    def forward(self, features):
        return self.topology.unpool(features, self.reduce)


class LinearDeformationBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        linear_channels: list[int],
        out_channels: int,
        batch_norm: bool = False,
        # add_normal_features=False,
    ) -> None:
        super().__init__()
        """Quadrature deformation block.

        Parameters
        ----------
        in_channels : int
        out_channels : int
        channels :
        batch_norm: bool

        Returns
        -------


        """
        # self.add_normal_features = add_normal_features
        # self.n_steps = n_steps
        # self.step_size = 1.0 / self.n_steps

        # if self.add_normal_features:
        #     assert topology is not None
        #     # initialize with batch size = 1
        #     self._surface = TemplateSurfaces(
        #         torch.empty((1, topology.n_vertices, 3), device=topology.faces.device),
        #         topology.faces,
        #     )
        #     in_channels += 3

        # we might as well use linear layers but use 1D convolutions instead as
        # that expects tensors of the format (N, C)

        for out_ch in linear_channels:
            self.add_module("Convolution", torch.nn.Conv1d(in_channels, out_ch, 1))
            if batch_norm:
                self.add_module.append("BatchNorm", torch.nn.BatchNorm1d(out_ch))
            self.add_module("Activation", torch.nn.PReLU())
            in_channels = out_ch

        # Final convolution to predict deformation vector
        self.add_module(
            "Convolution[out]", torch.nn.Conv1d(in_channels, out_channels, 1)
        )


class ConvolutionRepeater(torch.nn.Sequential):
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
