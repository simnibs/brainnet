import torch

from brainnet.mesh import topology
from brainnet.modules.graph import layers


class SurfaceModule(torch.nn.Module):
    def __init__(
        self,
        in_order: int = 0,
        out_order: int = 6,
        max_order: int = 6,  # n_topologies: int = 7, # 0 - n_topologies
        white_kwargs: dict | None = None,
        pial_kwargs: dict | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # TOPOLOGIES
        # assert n_topologies >= out_res + 1
        assert max_order >= out_order
        self.out_order = out_order

        # The topology is defined on the left hemisphere and although the
        # topology is the same for both hemispheres, we need to reverse the
        # order of the vertices in face array in order for the ordering to
        # remain consistent (e.g., counter-clockwise) once the vertices are
        # (almost) left-right mirrored

        # We use the left topology in the submodules which only use knowledge
        # of the neighborhoods to define the convolutions (and this is
        # independent of the face orientation).

        # self.topologies = topology.get_recursively_subdivided_topology(
        #     max_order,
        #     device=self.device,
        # )

        self.topologies = topology.StandardTopology.recursive_subdivision(
            max_order,
            device=self.device,
        )

        # self.topologies = topology.get_fsaverage_topology(max_order, self.device)

        # self.topologies = topology.FsAverageTopology.recursive_subdivision(
        #     max_order,
        #     device=self.device,
        # )

        self.active_topologies = list(range(in_order, max_order + 1))

        self.out_topology = self.topologies[self.active_topologies[-1]]

        # WHITE MATTER CONFIG
        if white_kwargs is None:
            white_kwargs = {}
        if "n_steps" not in white_kwargs:
            white_kwargs["n_steps"] = dict(
                zip(self.active_topologies, [2, 2, 2, 2, 2, 2, 1])
            )
        if "feature_maps" not in white_kwargs:
            white_kwargs["feature_maps"] = dict(
                zip(
                    self.active_topologies,
                    (
                        ["encoder:3", "decoder:0"],
                        ["encoder:2", "decoder:1"],
                        ["encoder:1", "decoder:2"],
                        ["encoder:0", "decoder:3"],
                        ["encoder:0", "decoder:3"],
                        ["encoder:0", "decoder:3"],
                        ["encoder:0", "decoder:3"],
                    ),
                )
            )

        # PIAL CONFIG
        if pial_kwargs is None:
            pial_kwargs = {}
        if "n_steps" not in pial_kwargs:
            pial_kwargs["n_steps"] = 10
        if "feature_maps" not in pial_kwargs:
            pial_kwargs["feature_maps"] = ["encoder:0", "decoder:3"]

        self.white_n_steps = white_kwargs["n_steps"]
        self.white_step_size = {k: 1.0 / v for k, v in self.white_n_steps.items()}
        self.white_feature_maps = white_kwargs["feature_maps"]

        self.pial_n_steps = pial_kwargs["n_steps"]
        self.pial_step_size = 1.0 / self.pial_n_steps
        self.pial_feature_maps = pial_kwargs["feature_maps"]

        self.white_deform = torch.nn.ModuleDict()
        self.pial_deform = torch.nn.Module()

    def forward(
        self,
        features: dict[str, torch.Tensor],
        template_vertices: dict[str, torch.Tensor],
    ):
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
        # features = [features] if isinstance(features, torch.Tensor) else features

        # The last feature map has the same spatial dimensions as the input
        # image
        self.set_image_center(features["decoder:3"])

        return {
            h: self._forward_hemi(features, v) for h, v in template_vertices.items()
        }

    def _forward_hemi(self, features: dict[str, torch.Tensor], vertices: torch.Tensor):
        """Predict placement of white matter surface and pial surface.."""

        # (N, M, 3) -> (N, 3, M) such that coordinates are in the channel
        # (feature) dimension and normalize
        vertices = vertices.mT

        white_vertices = self._estimate_white(features, vertices)
        pial_vertices = self._esimate_pial(features, white_vertices)

        white_vertices = white_vertices.mT
        pial_vertices = pial_vertices.mT

        # Transpose back to (N, M, 3)
        return dict(white=white_vertices, pial=pial_vertices)

    def _get_features(self, features, maps):
        return torch.cat([features[m] for m in maps], dim=1)

    def _estimate_white(self, features: dict[str, torch.Tensor], v: torch.Tensor):
        for order in self.active_topologies:
            step_size = self.white_step_size[order]
            deform = self.white_deform[str(order)]
            fmap = self._get_features(features, self.white_feature_maps[order])
            for _ in range(self.white_n_steps[order]):
                v_features = self.grid_sample(fmap, v)
                v = v + step_size * deform(v_features)
            if order < self.out_order:
                v = self.topologies[order].subdivide_vertices(v)
        return v

    def _esimate_pial(self, features: dict[str, torch.Tensor], v: torch.Tensor):
        fmap = self._get_features(features, self.pial_feature_maps)
        for _ in range(self.pial_n_steps):
            v_features = self.grid_sample(fmap, v)
            v = v + self.pial_step_size * self.pial_deform(v_features)
        return v

    def grid_sample_features(
        self, features: list[torch.Tensor], vertices: torch.Tensor
    ):
        return torch.cat([self.grid_sample(f, vertices) for f in features], dim=1)

    def grid_sample(self, image, vertices):
        """

        Parameters
        ----------
        image :
            image shape is (N, C, W, H, D)
        vertices :
            vertices shape is (N, 3, M)

        Returns
        -------
        samples :
            samples shape (N, C, M)
        """

        # vertices are in voxel coordinates

        # Transform vertices from (0, shape) to (-half_shape, half_shape), then
        # normalize to [-1, 1]
        # half_shape = (torch.as_tensor(image.shape[-3:], device=image.device) - 1) / 2
        # points = (vertices.mT - half_shape) / half_shape # N,3,M -> N,M,3

        # points = vertices.mT
        vertices = self.normalize_coordinates(vertices)

        # samples is N,C,D,H,W where C is from `image` and D,H,W are from `points`
        samples = torch.nn.functional.grid_sample(
            image.swapaxes(2, 4),  # N,C,W,H,D -> N,C,D,H,W
            # N,3,M -> N,M,3 -> N,D,H,W,3 where D=M; H=W=1
            vertices.mT[:, :, None, None],
            align_corners=True,
        )
        return samples[..., 0, 0]  # squeeze out H, W

    def set_image_center(self, image):
        """This is used with grid sampling when align_corners=True."""
        self._image_shape = torch.tensor(image.shape[-3:], device=image.device)
        center = 0.5 * (self._image_shape - 1.0)
        self._image_center = center[None, :, None]

    def normalize_coordinates(self, coords):
        # vertices are in voxel coordinates

        # Transform vertices from (0, shape) to (-half_shape, half_shape), then
        # normalize to [-1, 1]
        return (coords - self._image_center) / self._image_center

    def unnormalize_coordinates(self, coords):
        return self._image_center * coords + self._image_center


def make_unet_channels(in_channels: int, depth: int, multiplier: int = 2) -> dict:
    """Construct Unet hierarchy"""

    assert depth >= 1
    m = depth - 1
    # encoder = [in_channels * multiplier**i for i in range(m)]
    # ubend = in_channels * multiplier**m
    # return dict(encoder=encoder, ubend=ubend, decoder=decoder)
    encoder = [in_channels * multiplier**i for i in range(m + 1)]
    decoder = encoder[:-1][::-1]
    return dict(encoder=encoder, decoder=decoder)


class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        topologies: list[topology.Topology],
        conv_module: torch.nn.Module,
        reduce: str = "amax",
        channels: int | dict = 32,
        max_depth: int = 4,
        n_conv: int = 1,
    ):
        """Similar to a conventional UNet achitecture but on a graph. We
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
        max_depth: int
        multiplier: int
        n_conv: int


        """
        # The UNet architecture and naming
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

        max_depth = min(max_depth, len(topologies))
        self.topologies = topologies[-max_depth:]

        unet_channels = (
            make_unet_channels(channels, max_depth)
            if isinstance(channels, int)
            else channels
        )
        assert isinstance(unet_channels, dict)

        # Encoder
        in_ch = in_channels

        enc_topologies = self.topologies[1:][::-1]
        # get only the first channels if `topologies` is smaller than unet levels
        enc_channels = unet_channels["encoder"][::-1][: max_depth - 1][::-1]

        self.encoder_conv = torch.nn.ModuleList()
        self.encoder_pool = torch.nn.ModuleList()
        for out_ch, topo in zip(enc_channels, enc_topologies):
            self.encoder_conv.append(
                layers.nConv(in_ch, out_ch, conv_module, topo, n_conv)
            )
            self.encoder_pool.append(layers.Pool(topo, reduce))
            in_ch = out_ch

        # U bend
        self.ubend_conv = layers.nConv(
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
        dec_channels = unet_channels["decoder"][: max_depth - 1]

        self.decoder_unpool = torch.nn.ModuleList()
        self.decoder_conv = torch.nn.ModuleList()
        for out_ch, skip_ch, unpool_topology, conv_topology in zip(
            dec_channels, skip_channels, unpool_topologies, conv_topologies
        ):
            self.decoder_unpool.append(layers.Unpool(unpool_topology, reduce))
            self.decoder_conv.append(
                layers.nConv(
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
            features = unpool(features)
            features = torch.concat((features, sf), dim=1)
            features = conv(features)

        return features


class UNetTransform(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        topologies: list[topology.Topology],
        channels: None | dict = None,
        unet_conv_module: torch.nn.Module = layers.EdgeConvolutionBlock,
        deform_conv_module: torch.nn.Module = layers.EdgeConvolution,
        reduction: str = "amax",
        max_depth: int = 4,
        n_convolutions: int = 1,
    ) -> None:
        """This graph deformation block uses a graph UNet to extract features
        which are transformed to deformation vectors. These are then applied to
        the vertices at whose positions the features were estimated.
        """
        super().__init__()

        # # Initialize this way to get parameter._version == 1 like all the rest
        # self.n_steps = n_steps
        # self.step_size = torch.nn.Parameter(torch.empty([1]))
        # torch.nn.init.constant_(self.step_size, 1.0 / n_steps)

        if channels is None:
            channels = dict(encoder=[64, 96, 128], ubend=160, decoder=[128, 96, 64])

        unet = UNet(
            in_channels,
            topologies,
            unet_conv_module,
            reduce=reduction,
            channels=channels,
            max_depth=max_depth,
            n_conv=n_convolutions,
        )

        # Final convolution block to estimate deformation field from features
        reduce_index, gather_index = topologies[-1].get_convolution_indices()
        deform = deform_conv_module(
            unet.out_ch,
            out_channels,
            reduce_index,
            gather_index,
            bias=False,
            init_zeros=True,
        )
        self.transform = torch.nn.Sequential(unet, deform)

    def forward(self, features):
        """Apply graph UNet and estimate deformation vectors from the
        resulting features. A forward Euler scheme is used to move the vertices
        to their new locations by scaling the deformation vectors before they
        are applied to the vertices. This process is repeat
        `self.euler_iterations` number of times.
        """
        # for _ in torch.arange(self.n_steps):
        #     sampled_f = grid_sample(features, vertices)  # image features
        #     sampled_f = self.unet(sampled_f)  # graph features
        #     dV = self.spatial_deform(sampled_f)
        #     vertices = vertices + self.step_size * dV

        return self.transform(features)


class LinearTransform(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: list[int],
        batch_norm=False,
        # add_normal_features=False,
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
        self.transform = torch.nn.Sequential()
        for out_ch in channels:
            self.transform.append(torch.nn.Conv1d(in_channels, out_ch, 1))
            if batch_norm:
                self.transform.append(torch.nn.BatchNorm1d(out_ch))
            self.transform.append(torch.nn.PReLU())
            in_channels = out_ch

        # Final convolution to predict deformation vector
        self.transform.append(torch.nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, features):
        """

        image :

        vertices :
            shape (N, M, 3)
        """
        # for _ in torch.arange(self.n_steps, device=image.device):
        #     features = grid_sample(image, vertices)

        #     # if self.add_normal_features:
        #     #     self._surface.vertices = vertices.mT
        #     #     nn = self._surface.compute_vertex_normals().mT
        #     #     features = torch.cat((features, nn), dim=1)

        #     vertices = vertices + self.step_size * self.transform(features)

        # return vertices
        return self.transform(features)


# class TopoFitTemplateInitialization(torch.nn.Module):
#     def __init__(self, in_channels, n_points) -> None:
#         super().__init__()

#         conv_kwargs = dict(spatial_dims=3)
#         self.conv = Convolution(
#             in_channels=in_channels, out_channels=n_points, **conv_kwargs
#         )

#     def cache_shape_and_grid(self, shape, device):
#         """During training the shape will stay the same so no need to recreate
#         the grid on every forward pass.
#         """
#         if not hasattr(self, "shape") or shape != self.shape:
#             self.shape = shape
#             self.grid = torch.meshgrid(
#                 [torch.arange(s, device=device) for s in shape], indexing="ij"
#             )
#             self.grid = tuple(g.ravel() for g in self.grid)

#             # self.register_buffer("grid", persistent=False)

#     def forward(self, features, eps=1e-6):
#         """Predict the coordinates of a number of points.

#         Parameters
#         ----------
#         features : (N, C, ...)

#         Returns
#         -------
#         vertices : (N, 3, M)
#             M is the number of predict vertices.
#         """
#         N = features.shape[0]
#         shape = features.shape[2:]

#         self.cache_shape_and_grid(shape, features.device)

#         x = self.conv(features)
#         x = x.reshape(N, self.conv.out_channels, -1)
#         x = x / (x.sum(-1)[..., None] + eps)  # normalize channel maps

#         # weigh coordinates (grid) by feature maps
#         return torch.stack([torch.sum(x * g.ravel(), dim=-1) for g in self.grid], dim=1)


# class TopoFitGraphAdjust(torch.nn.Module):
#     def __init__(
#         self,
#         surfaces,
#         in_channels,
#         resolution: int = 6,
#         channels: list[int] = [32],
#         n_iterations: int = 5,
#         device: str = "cpu",
#     ) -> None:
#         super().__init__()

#         self.prediction_topology = topology.get_recursively_subdivided_topology(
#             resolution,
#             topology.initial_faces.to(device),
#         )[-1]

#         self.deformation = torch.nn.ModuleDict(
#             {
#                 k: layers.GraphLinearDeform(
#                     in_channels, channels, n_iterations, topology=self.prediction_topology
#                 )
#                 for k in surfaces
#             }
#         )

#     def get_prediction_topology(self):
#         return self.prediction_topology

#     def forward(
#         self, features: torch.Tensor, initial_vertices: dict[str, dict[str, torch.Tensor]],
#     ):
#         # initial_vertices, e.g.,
#         #   {lh: {white: tensor, pial: tensor}, rh: {white: tensor, pial:tensor}}
#         return {
#             hemi: {
#                 surf: self.deformation[surf](features, v.mT).mT for surf, v in surfs.items()
#             }
#             for hemi, surfs in initial_vertices.items()
#         }
