import torch

from brainnet.modules.topofit.utilities import grid_sample
from brainnet.mesh.surface import TemplateSurfaces


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
        bias=False,
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

        self.batchnorm = torch.nn.BatchNorm1d(out_channels)

        # if activation is None:
        #     self.activation = lambda x: x
        # elif activation == "leaky":
        #     self.activation = torch.nn.PReLU(init=0.3)
        # else:
        #     raise ValueError
        self.activation = torch.nn.PReLU(init=0.3)

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

        out = self.batchnorm(out)

        return self.activation(out)


class EdgeConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduce_index: torch.Tensor,
        gather_index: torch.Tensor,
        # batchnorm=True,
        bias: bool = True,
        init_zeros: bool = False,
    ):
        super().__init__()
        self.in_channels = 2 * in_channels  # vertex and neighbor concatenation
        self.out_channels = out_channels
        self.reduce_index = reduce_index.long()  # repeats of vertex
        self.gather_index = gather_index.long()  # neighbors of vertex

        # self.linear = torch.nn.Linear(in_channels, out_channels, bias)
        self.conv = torch.nn.Conv1d(self.in_channels, out_channels, 1, bias=bias)

        if init_zeros:
            torch.nn.init.zeros_(self.conv.weight)
            if bias:
                torch.nn.init.zeros_(self.conv.bias)

        self.activation = torch.nn.PReLU(init=0.3)

        # self.batchnorm = torch.nn.BatchNorm1d(out_channels) if batchnorm else None

        # self.apply_dropout = dropout_p > 0.0
        # if self.apply_dropout:
        #     self.dropout = torch.nn.Dropout1d(dropout_p)

    def forward(self, in_features):
        # F_in : (batch, channels, vertices)
        batch_size, _, n_vertices = in_features.shape
        out_shape = batch_size, self.out_channels, n_vertices

        vertices = in_features[..., self.reduce_index]
        neighbors = in_features[..., self.gather_index]

        concat_features = torch.cat([vertices, neighbors - vertices], dim=1)
        F_e = self.conv(concat_features)

        # Index pooling of features
        out = torch.zeros(out_shape, dtype=in_features.dtype, device=in_features.device)
        out.index_reduce_(
            dim=-1,
            index=self.reduce_index,
            source=F_e,
            reduce="mean",
            include_self=False,
        )

        # if self.batchnorm is not None:
        #     out = self.batchnorm(out)

        return self.activation(out)


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


class GraphLinearDeform(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        n_iterations: int = 1,
        batch_norm = False,
        add_normal_features = False,
        topology=None,
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
        ndim = 3

        self.add_normal_features = add_normal_features
        self.n_iterations = n_iterations
        self.scale = 1 / self.n_iterations

        if self.add_normal_features:
            assert topology is not None
            # initialize with batch size = 1
            self._surface = TemplateSurfaces(
                torch.empty((1, topology.n_vertices, 3), device=topology.faces.device),
                topology.faces,
            )

            in_channels += 3

        # we might as well use linear layers but use 1D convolutions instead as
        # that expects tensors of the format (N, C)
        self.quad_block = torch.nn.Sequential()
        for out_ch in channels:
            self.quad_block.append(torch.nn.Conv1d(in_channels, out_ch, 1))
            if batch_norm:
                self.quad_block.append(torch.nn.BatchNorm1d(out_ch))
            self.quad_block.append(torch.nn.PReLU(init=0.3))
            in_channels = out_ch

        # we work in 3D so three output channels
        self.quad_block.append(torch.nn.Conv1d(in_channels, ndim, 1))

    def forward(self, image, vertices):
        """

        image :

        vertices :
            shape (N, M, 3)
        """
        for _ in torch.arange(self.n_iterations, device=image.device):
            features = grid_sample(image, vertices)

            if self.add_normal_features:
                self._surface.vertices = vertices.mT
                nn = self._surface.compute_vertex_normals().mT
                features = torch.cat((features, nn), dim=1)

            features = self.quad_block(features)
            vertices = vertices + self.scale * features

        return vertices
