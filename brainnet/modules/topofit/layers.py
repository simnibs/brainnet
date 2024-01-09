from typing import Union

import torch

from brainnet.mesh.topology import Topology
from brainnet.modules.topofit.utilities import grid_sample

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
        # bias=True,
        activation="leaky",
        # dropout_p=0.0,
        # init="normal",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduce_index = reduce_index.long()  # repeats of vertex
        self.gather_index = gather_index.long()  # neighbors of vertex

        # self.linear0 = torch.nn.Linear(in_channels, out_channels, bias=False)
        # self.linear1 = torch.nn.Linear(in_channels, out_channels, bias)

        self.conv0 = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)

        # self.batchnorm = torch.nn.BatchNorm1d(out_channels)

        if activation is None:
            self.activation = lambda x: x
        elif activation == "leaky":
            self.activation = torch.nn.LeakyReLU(0.3)
        else:
            raise ValueError

        # self.apply_dropout = dropout_p > 0.0
        # if self.apply_dropout:
        #     self.dropout = torch.nn.Dropout1d(dropout_p)

        # if init == "normal":
        #     torch.nn.init.normal_(self.linear0.weight, mean=0, std=0.01)
        #     torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
        #     self.linear0.bias.data.zero_()
        #     self.linear1.bias.data.zero_()
        # elif init == "zero":
        #     self.linear0.weight.data.zero_()
        #     self.linear1.weight.data.zero_()
        # else:
        #     raise ValueError

    def forward(self, in_features):
        # F_in : (batch, channels, vertices)

        # F_v = self.linear0(in_features)  # feature for vertex as center
        # F_n = self.linear1(in_features)  # feature for vertex as neighbor

        F_v = self.conv0(in_features)  # feature for vertex as center
        F_n = self.conv1(in_features)  # feature for vertex as neighbor

        # What we want is simply...
        #
        # for i,j in zip(self.scatter_index, self.gather_index):
        #     F_v[i] += F_N[j]
        #
        # and then mean...
        # F_v.scatter_reduce(1, self.reduce_index, F_N[:, self.gather_index], reduce="mean")


        # aggregate (neighbors): mean
        out = torch.zeros_like(F_v)
        out.index_reduce_(
            -1,
            self.reduce_index,
            F_n[..., self.gather_index],
            reduce="mean",
            include_self=False,
        )
        # merge (v and neighbors): sum
        out = out + F_v

        # F_v = self.batchnorm(F_v)

        return self.activation(out)


class EdgeConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduce_index: torch.Tensor,
        gather_index: torch.Tensor,
        activation="leaky",
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = 2 * in_channels  # vertex and neighbor concatenation
        self.out_channels = out_channels
        self.reduce_index = reduce_index.long()  # repeats of vertex
        self.gather_index = gather_index.long()  # neighbors of vertex

        # self.linear = torch.nn.Linear(in_channels, out_channels, bias)
        self.conv = torch.nn.Conv1d(self.in_channels, out_channels, 1, bias=bias)

        if activation is None:
            self.activation = lambda x: x
        elif activation == "leaky":
            self.activation = torch.nn.LeakyReLU(0.3)
        else:
            raise ValueError

        # self.apply_dropout = dropout_p > 0.0
        # if self.apply_dropout:
        #     self.dropout = torch.nn.Dropout1d(dropout_p)

    def forward(self, in_features):
        # F_in : (batch, channels, vertices)
        batch_size, _, n_vertices = in_features.shape
        out_shape = batch_size, self.out_channels, n_vertices

        vertices = in_features[..., self.reduce_index]
        neighbors = in_features[..., self.gather_index]

        concat_features = torch.cat([vertices, neighbors - vertices], dim=-2)
        # F_e = self.linear(concat_features)
        F_e = self.conv(concat_features)

        # transpose (B, V, C) to (B, C, V) for convolution layer and back
        # F_e = self.conv1d(concat_features.transpose(1, 2)).transpose(1, 2)

        # if self.apply_dropout:
        #     edge_features = self.dropout(edge_features)

        # edge_features_weighted = edge_features * self.weights
        # indices = self.edges_a.unsqueeze(-1).expand(-1, self.out_channels)

        # out_features = torch.zeros((self.size, self.out_channels), device=self.device)
        # out_features.scatter_reduce_(
        #     1,
        #     self.reduce_index[None, :, None].expand_as(edge_features),
        #     edge_features,
        #     reduce="mean",
        #     include_self=False, # self is already included
        # )

        # Index pooling of features
        out_features = torch.zeros(
            out_shape, dtype=in_features.dtype, device=in_features.device
        )
        out_features.index_reduce_(
            -1,
            self.reduce_index,
            F_e,
            reduce="mean",
            include_self=False,
        )
        return self.activation(out_features)


class NConv(torch.nn.Sequential):
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

        for i in torch.arange(n):
            self.add_module(
                f"conv{i}",
                conv_module(in_channels, out_channels, reduce_index, gather_index),
            )
            in_channels = out_channels


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


class GraphEncoderUnit(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        topology: Topology,
        conv_module: torch.nn.Module,
        reduce: str = "amax",
        n_conv: int = 1,
        pool: bool = True,
    ) -> None:
        """A single"""
        super().__init__()
        self.add_module(
            "nconv",
            NConv(in_channels, out_channels, conv_module, topology, n_conv),
        )
        if pool:
            self.add_module("pool", GraphPool(topology, reduce))


class GraphDecoderUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        topology: Topology,
        conv_module: torch.nn.Module,
        reduce: str = "amax",
        n_conv: int = 1,
        unpool: bool = True,
    ) -> None:
        """
        conv_module : torch.nn.Module
            E.g., GraphConv or EdgeConv
        """
        super().__init__()
        self.nconv = NConv(in_channels, out_channels, conv_module, topology, n_conv)
        self.unpool = GraphUnpool(topology, reduce) if unpool else lambda x: x

    @staticmethod
    def concatenate(a: torch.Tensor, b: torch.Tensor, dim: int):
        return torch.cat((a, b), dim)

    def forward(self, a: torch.Tensor, b: Union[None,torch.Tensor] = None):
        """Concatenate `a` with `b`, feed through convolutional layer(s), and
        unpool.
        """
        a = a if b is None else self.concatenate(a, b, dim=-2)
        return self.unpool(self.nconv(a))


class GraphLinearDeform(torch.nn.Module):
    def __init__(
        self, in_channels: int, channels: list[int], n_iterations: int = 1
    ) -> None:
        super().__init__()
        """Quadrature deformation block.

        Parameters
        ----------
        in_channels : int

        channels :

        n_iterations : int


        Returns
        -------


        """
        self.n_iterations = n_iterations
        self.scale = 1.0 / n_iterations

        self.quad_block = torch.nn.Sequential()
        for i, out_ch in enumerate(channels):
            # self.quad_block.add_module(
            #     f"Linear{i}", torch.nn.Linear(in_channels, out_ch)
            # )
            self.quad_block.add_module(f"Conv1d{i}", torch.nn.Conv1d(in_channels, out_ch, 1))
            self.quad_block.add_module(f"Activation{i}", torch.nn.LeakyReLU(0.3))
            in_channels = out_ch

        # self.quad_block.add_module(f"LinearDeform", torch.nn.Linear(in_channels, 3))
        self.quad_block.add_module(f"ConvDeform", torch.nn.Conv1d(in_channels, 3, 1))

    def forward(self, image, vertices):
        """

        image :

        vertices :
            shape (N, M, 3)
        """
        for _ in torch.arange(self.n_iterations):
            features = grid_sample(image, vertices)
            features = self.quad_block(features)
            vertices = vertices + self.scale * features
        return vertices
