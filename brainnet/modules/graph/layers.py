from typing import Type

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
        bias: bool = True,
        init_values: float | list[float] | None | tuple = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # repeats of vertex
        self.register_buffer("reduce_index", reduce_index.long(), persistent=False)
        # neighbors of vertex
        self.register_buffer("gather_index", gather_index.long(), persistent=False)

        kwargs = dict(kernel_size=1, stride=1, bias=bias)

        # Compute features when a vertex is the position of interest
        self.conv_self = torch.nn.Conv1d(in_channels, out_channels, **kwargs)
        # Compute features when a vertex is part of the neighborhood
        self.conv_other = torch.nn.Conv1d(in_channels, out_channels, **kwargs)

        if init_values is not None:
            self.init_with_values(self.conv_self.weight, init_values)
            self.init_with_values(self.conv_other.weight, init_values)
            if bias:
                self.init_with_values(self.conv_self.bias, init_values)
                self.init_with_values(self.conv_other.bias, init_values)

    def init_with_values(self, weights, values):
        if isinstance(values, float):
            torch.nn.init.constant_(weights, values)
            torch.nn.init.constant_(weights, values)
        else:
            assert len(values) == self.out_channels
            for w, value in zip(weights, values):
                torch.nn.init.constant_(w, value)

    def forward(self, features):
        features_self = self.conv_self(features)  # W0
        features_other = self.conv_other(features)  # W1

        # aggregate (neighbors)
        out = torch.index_add(
            torch.zeros_like(features_self),
            -1,
            self.reduce_index,
            features_other[..., self.gather_index],
        )

        # merge (self and neighbors)
        return features_self + out


class EdgeConvolution(GraphConvolution):
    def forward(self, features):
        features_self = self.conv_self(features)  # W0
        features_other = self.conv_other(features)  # W1

        out = torch.index_reduce(
            torch.zeros_like(features_other),
            -1,
            self.reduce_index,
            features_other[..., self.gather_index],
            "mean",
            include_self=False,
        )
        return features_self + out - features_other


def convolution_block(Convolution: Type[torch.nn.Module]):
    class ConvolutionBlock(Convolution):
        """Graph convolution followed by normalization, activation, and dropout."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reduce_index: torch.Tensor,
            gather_index: torch.Tensor,
            bias: bool = True,
            init_values: float | list[float] | None | tuple = None,
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
                init_values,
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
        self.convs.append(
            GraphConvolutionBlock(channels[-2], channels[-1], activation=None, **kwargs)
        )
        self.norm = torch.nn.InstanceNorm1d(
            channels[0], affine=False, track_running_stats=False
        )
        self.activation = torch.nn.PReLU()

    def forward(self, features):
        return self.activation(self.convs(features) + self.norm(features))


class GraphConvolutionDeformationBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        graph_channels: list[int],
        out_channels: int,
        reduce_index,
        gather_index,
        out_init_values: float | list[float] | None | tuple = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index, **kwargs)
        # input features -> features
        # features -> features
        for i, (in_ch, out_ch) in enumerate(
            zip([in_channels] + graph_channels[:-1], graph_channels)
        ):
            self.add_module(
                f"GraphConvolution_{i}", GraphConvolutionBlock(in_ch, out_ch, **kw)
            )
        # features -> deformation field
        self.add_module(
            f"GraphConvolution_{i+1}",
            GraphConvolution(
                graph_channels[-1], out_channels, init_values=out_init_values, **kw
            ),
        )


class EdgeConvolutionDeformationBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        graph_channels: list[int],
        out_channels: int,
        reduce_index,
        gather_index,
        out_init_values: float | list[float] | None | tuple = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index, **kwargs)
        # input features -> features
        # features -> features
        for i, (in_ch, out_ch) in enumerate(
            zip([in_channels] + graph_channels[:-1], graph_channels)
        ):
            self.add_module(
                f"GraphConvolution_{i}", EdgeConvolutionBlock(in_ch, out_ch, **kw)
            )
        # features -> deformation field
        self.add_module(
            f"GraphConvolution_{i+1}",
            EdgeConvolution(
                graph_channels[-1], out_channels, init_values=out_init_values, **kw
            ),
        )


class ResidualGraphConvolutionDeformationBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        graph_channels: list[int],
        out_channels: int,
        reduce_index,
        gather_index,
        n_residual_blocks: int = 3,
        out_init_values: float | list[float] | None | tuple = 0.0,
    ) -> None:
        super().__init__()
        kw = dict(reduce_index=reduce_index, gather_index=gather_index)
        # input features -> features
        self.add_module(
            "GraphConvolution_0", GraphConvolution(in_channels, graph_channels[0], **kw)
        )
        # features -> features
        for i in range(n_residual_blocks):
            self.add_module(
                f"ResidualGraphConvolution_{i}",
                ResidualGraphConvolution(graph_channels, **kw),
            )
        # features -> deformation field
        self.add_module(
            "GraphConvolution_1",
            GraphConvolution(
                graph_channels[-1], out_channels, init_values=out_init_values, **kw
            ),
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
        out_init_values: float | list[float] | None | tuple = None,
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
        # we might as well use a linear layer but use 1D conv instead as that
        # expects tensors of the format (N, C)
        self.out_channels = out_channels

        for i, out_ch in enumerate(linear_channels):
            self.add_module(f"Convolution_{i}", torch.nn.Conv1d(in_channels, out_ch, 1))
            if batch_norm:
                self.add_module.append(f"BatchNorm_{i}", torch.nn.BatchNorm1d(out_ch))
            self.add_module(f"Activation_{i}", torch.nn.PReLU())
            in_channels = out_ch

        # Final convolution to predict deformation vector
        self.add_module(
            f"Convolution_{i+1}", torch.nn.Conv1d(in_channels, out_channels, 1)
        )

        if out_init_values is not None:
            self.init_with_values(self[-1].weight, out_init_values)
            self.init_with_values(self[-1].bias, out_init_values)

    def init_with_values(self, weights, values):
        if isinstance(values, float):
            torch.nn.init.constant_(weights, values)
            torch.nn.init.constant_(weights, values)
        else:
            assert len(values) == self.out_channels
            for w, value in zip(weights, values):
                torch.nn.init.constant_(w, value)


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

        for _ in torch.arange(n):
            self.append(
                conv_module(
                    in_channels,
                    out_channels,
                    topology.conv_index_reduce,
                    topology.conv_index_gather,
                ),
            )
            in_channels = out_channels
