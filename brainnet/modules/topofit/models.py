from copy import deepcopy
from typing import Union

import torch
from monai.networks.blocks import Convolution

from brainnet.modules.topofit.config import TopoFitModelParameters, UnetParameters

from brainnet.modules.topofit import layers
from brainnet.modules.topofit.utilities import grid_sample
from brainnet.mesh import topology
from brainnet.mesh.surface import BatchedSurfaces

default_hemispheres = ("lh", "rh")


def make_unet_channels(in_channels: int, n_levels: int, multiplier: int) -> dict:
    """Construct Unet hierarchy"""

    assert n_levels >= 1
    m = n_levels - 1
    encoder = [in_channels * multiplier**i for i in range(m)]
    ubend = in_channels * multiplier**m
    decoder = encoder[::-1]
    return dict(encoder=encoder, ubend=ubend, decoder=decoder)


class GraphUnet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        topologies: list[topology.Topology],
        conv_module: Union[layers.GraphConv, layers.EdgeConv],
        reduce: str = "amax",
        channels: Union[int, dict] = 32,
        n_levels: int = 4,
        multiplier: int = 2,
        n_conv: int = 1,
    ):
        """Similar to a conventional U-net achitecture but on a graph. We
        exploit the fact that the topologies represent a hierarchy of
        recursive subdivision. Consequently, we can move up and down this
        hierarchy to obtain different mesh resolutions.


        Parameters
        ----------
        in_channels : int
        topologies :
        conv_module
        reduce: str
        channels: int | dict
        n_levels: int
        multiplier: int
        n_conv: int


        """
        # The U-net architecture and naming
        #
        # ENCODER                         DECODER   Hierarchy level (example)
        #
        # I C C ------------------------- I C C     4
        #     P                           U
        #     I C C ----------------- I C C         3
        #         P                   U
        #         I C C --------- I C C             2
        #             P           U
        #             I C C - I C C                 1
        #                 P   U
        #                 I C C                     0
        #                 U-bend
        #
        # I : input
        # C : conv
        # P : pooling
        # U : unpooling
        # - : skip connection
        #
        # Encoder unit: (I-)C-C-P
        # Decoder unit: U(-I)-C-C
        # U-bend: (I-)C-C
        super().__init__()

        n_levels = min(n_levels, len(topologies))
        self.topologies = topologies[-n_levels:]

        unet_channels = (
            make_unet_channels(channels, n_levels, multiplier)
            if isinstance(channels, int)
            else channels
        )
        assert isinstance(unet_channels, dict)

        self.encoder_conv = torch.nn.ModuleList()
        self.encoder_pool = torch.nn.ModuleList()
        self.decoder_conv = torch.nn.ModuleList()
        self.decoder_unpool = torch.nn.ModuleList()

        # Encoder
        in_ch = in_channels

        enc_topologies = self.topologies[1:][::-1]
        # get only the first channels if `topologies` is smaller than unet levels
        enc_channels = unet_channels["encoder"][::-1][: n_levels - 1][::-1]
        for out_ch, topology in zip(enc_channels, enc_topologies):
            self.encoder_conv.append(
                layers.NConv(in_ch, out_ch, conv_module, topology, n_conv)
            )
            self.encoder_pool.append(layers.GraphPool(topology, reduce))

            in_ch = out_ch

        # U bend
        self.ubend_conv = layers.NConv(
            in_ch,
            out_ch := unet_channels["ubend"],
            conv_module,
            self.topologies[0],
            n_conv,
        )
        in_ch = out_ch

        # Decoder
        unpool_topologies = self.topologies[:-1]
        conv_topologies = self.topologies[1:]
        skip_channels = enc_channels[::-1]
        # get only the first channels if `topologies` is smaller than unet levels
        dec_channels = unet_channels["decoder"][: n_levels - 1]
        for out_ch, skip_ch, unpool_topology, conv_topology in zip(
            dec_channels, skip_channels, unpool_topologies, conv_topologies
        ):
            self.decoder_unpool.append(layers.GraphUnpool(unpool_topology, reduce))
            self.decoder_conv.append(
                layers.NConv(
                    in_ch + skip_ch, out_ch, conv_module, conv_topology, n_conv
                )
            )
            in_ch = out_ch
        self.out_ch = out_ch

    def get_prediction_topology(self):
        return self.topologies[-1]

    def forward(self, features):
        # Encoder
        skip_features = []
        for conv, pool in zip(self.encoder_conv, self.encoder_pool):
            features = conv(features)
            skip_features.append(features)
            features = pool(features)

        features = self.ubend_conv(features)

        # Decoder
        for conv, unpool, sf in zip(
            self.decoder_conv, self.decoder_unpool, skip_features[::-1]
        ):
            features = conv(torch.concat((unpool(features), sf), -2))

        return features


class GraphUnetDeform(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        topologies: list[topology.Topology],
        conv_module: Union[layers.GraphConv, layers.EdgeConv],
        euler_step_size: float,
        euler_iterations: int,
        config_unet: UnetParameters,
    ) -> None:
        """This graph deformation block uses a graph u-net to extract features
        which are transformed to deformation vectors which are then applied to
        the vertices at whose positions the features were estimated.
        """
        super().__init__()
        self.euler_step_size = euler_step_size
        self.euler_iterations = euler_iterations

        self.unet = GraphUnet(
            in_channels,
            topologies,
            getattr(layers, config_unet.conv_module),
            reduce=config_unet.reduction,
            channels=config_unet.channels,
            n_levels=config_unet.n_levels,
            multiplier=config_unet.multiplier,
            n_conv=config_unet.n_conv,
        )

        # Final convolution block to estimate deformation field from features
        (
            reduce_index,
            gather_index,
        ) = self.get_prediction_topology().get_convolution_indices()

        self.spatial_deform = conv_module(
            self.unet.out_ch, 3, reduce_index, gather_index, activation=None
        )

    def get_prediction_topology(self):
        return self.unet.get_prediction_topology()

    def forward(self, image, vertices):
        """Apply graph U-net and estimate deformation vectors from the
        resulting features. A forward Euler scheme is used to move the vertices
        to their new locations by scaling the deformation vectors before they
        are applied to the vertices. This process is repeat
        `self.euler_iterations` number of times.
        """
        for _ in torch.arange(self.euler_iterations):
            features = grid_sample(image, vertices)  # image features
            # possibility to add normalized vertex coordinates as features?
            # ...
            features = self.unet(features)  # graph features
            deformation = self.spatial_deform(features)
            vertices = vertices + self.euler_step_size * deformation

        return vertices


class TopoFitTemplateInitialization(torch.nn.Module):
    def __init__(self, in_channels, n_points) -> None:
        super().__init__()

        conv_kwargs = dict(spatial_dims=3)
        self.conv = Convolution(
            in_channels=in_channels, out_channels=n_points, **conv_kwargs
        )

    def cache_shape_and_grid(self, shape, device):
        """During training the shape will stay the same so no need to recreate
        the grid on every forward pass.
        """
        if not hasattr(self, "shape") or shape != self.shape:
            self.shape = shape
            self.grid = torch.meshgrid(
                [torch.arange(s, device=device) for s in shape], indexing="ij"
            )
            self.grid = tuple(g.ravel() for g in self.grid)

            # self.register_buffer("grid", persistent=False)

    def forward(self, features, eps=1e-6):
        """Predict the coordinates of a number of points.

        Parameters
        ----------
        features : (N, C, ...)

        Returns
        -------
        vertices : (N, 3, M)
            M is the number of predict vertices.
        """
        N = features.shape[0]
        shape = features.shape[2:]

        self.cache_shape_and_grid(shape, features.device)

        x = self.conv(features)
        x = x.reshape(N, self.conv.out_channels, -1)
        x = x / (x.sum(-1)[..., None] + eps)  # normalize channel maps

        # weigh coordinates (grid) by feature maps
        return torch.stack([torch.sum(x * g.ravel(), dim=-1) for g in self.grid], dim=1)


class TopoFitGraph(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        prediction_res: int = 6,
        n_template_vertices: int = 62,
        device="cpu",
        # image_shape: torch.IntTensor | torch.LongTensor,
        # config: None | config.TopoFitModelParameters = None,
    ) -> None:
        super().__init__()

        self.device = device

        config = TopoFitModelParameters

        n_res = len(config.unet_deform.resolutions)
        assert prediction_res <= config.unet_deform.resolutions[-1]
        assert n_res >= prediction_res + 1
        self.prediction_res = prediction_res

        # self.register_buffer(
        #     "initial_faces", topology.initial_faces, persistent=False,
        # )

        # We use the left topology in the submodules which only use knowledge
        # of the neighborhoods to define the convolutions (and this is
        # independent of the face orientation).
        self.topologies = topology.get_recursively_subdivided_topology(
            n_res - 1,
            topology.initial_faces.to(self.device),
        )

        # The topology is defined on the left hemisphere and although the
        # topology is the same for both hemispheres, we need to reverse the
        # order of the vertices in face array in order for the ordering to
        # remain consistent (e.g., counter-clockwise) once the vertices are
        # (almost) left-right mirrored

        # Add an initializer for the template positions
        for hemi in default_hemispheres:
            self.add_module(
                f"template_initialization_{hemi}",
                TopoFitTemplateInitialization(in_channels, n_template_vertices),
            )

        # Surface placement modules are shared for both hemispheres

        # white matter placement
        conv_module = getattr(layers, config.unet_deform.conv_module)
        options_gdb = zip(
            config.unet_deform.resolutions,
            config.unet_deform.euler_step_size,
            config.unet_deform.euler_iterations,
        )
        self.unet_deform = torch.nn.ModuleList(
            [
                GraphUnetDeform(
                    in_channels,
                    self.topologies[: res + 1],
                    conv_module,
                    euler_step_size,
                    euler_iterations,
                    config.unet_deform.unet,
                )
                for res, euler_step_size, euler_iterations in options_gdb
            ]
        )

        # layer 4 and pial placement
        # NOTE one or two linear deformation layers? shared or different parameters?
        self.linear_deform = layers.GraphLinearDeform(
            in_channels,
            config.linear_deform.channels,
            config.linear_deform.n_iterations,
        )
        # self.linear_deform1 = GraphLinearDeform(in_channels,  **kwargs_quad)

    def get_prediction_topology(self):
        return self.topologies[self.prediction_res]

    def forward(self, features: torch.Tensor, hemispheres: None | tuple | list = None):
        """
        Faces can be retrieved from

            faces = self.topologies[self.prediction_res].faces

        Parameters
        ----------
        features : torch.Tensor
            Tensor of shape (N, C, W, H, D) where N is batch size and C is the
            number of channels (feature maps).
            NOTE Torch assumes that images are (N,C,D,H,W). For convolutions
            this does not really matter, however, when we sample features for
            the surface vertices using `grid_sample`, we need to transpose D
            and W to that they correspond to the coordinates of `vertices`
            which are x,y,z (W,H,D).
        vertices : torch.Tensor
            Tensor of shape (N, M, 3) where M is the number of vertices and the
            last dimension contains the coordinates (x,y,z).

        Returns
        -------

        """
        hemispheres = default_hemispheres if hemispheres is None else hemispheres
        out = {}
        for hemi in hemispheres:
            # Position the initial template vertices

            # transpose to (N, 3, M) (.mT = batch transpose) such that coordinates
            # are in the channel dimension
            # vertices = vertices.mT
            vertices = getattr(self, f"template_initialization_{hemi}")(
                features
            )  # (N, 3, M)

            out[hemi] = self._topofit_forward(features, vertices)

        return out

    def _topofit_forward(self, features, vertices):
        """TopoFit prediction."""
        # At each iteration
        #   1. Sample image features at the corresponding points on the surface
        #   2. use these features to predict a deformation of each point

        # Place white matter
        for res in range(self.prediction_res + 1):
            # for i,unet_deform_block in enumerate(self.unet_deform):
            unet_deform = self.unet_deform[res]
            vertices = unet_deform(features, vertices)
            vertices = (
                unet_deform.get_prediction_topology().subdivide_vertices(vertices)
                if res < self.prediction_res
                else vertices
            )

        # Expand white matter

        # NOTE Currently layer4 is not supervised so it is not valid
        vertices_layer4 = self.linear_deform(features, vertices)
        vertices_pial = self.linear_deform(features, vertices_layer4)

        # Transpose back to (N, M, 3)
        return dict(white=vertices.mT, pial=vertices_pial.mT)
