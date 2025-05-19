import copy
import torch

import brainnet.mesh.topology
from brainnet.mesh.surface import load_deepsurfer_template, Surface
from brainnet.modules.graph import layers


class GenericSurfaceModule(torch.nn.Module):
    def __init__(
        self,
        in_order: int,
        out_order: int,
        max_order: int,  # n_topologies: int = 7, # 0 - n_topologies
        topology: str = "DeepSurferTopology",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.in_order = in_order
        self.max_order = max_order
        self.initialize_topologies(topology)
        self.set_out_order(out_order)

    def initialize_topologies(self, topology):
        # The DeepSurfer topology is defined on the left hemisphere and
        # although the topology is the same for both hemispheres, we need to
        # reverse the
        # order of the vertices in face array in order for the ordering to
        # remain consistent (e.g., counter-clockwise) once the vertices are
        # (almost) left-right mirrored

        # We use the left topology in the submodules which only use knowledge
        # of the neighborhoods to define the convolutions (and this is
        # independent of the face orientation).

        self.topologies = getattr(
            brainnet.mesh.topology, topology
        ).recursive_subdivision(
            self.max_order,
            device=self.device,
        )
        self.all_topologies = list(range(self.in_order, self.max_order + 1))
        self.n_topologies = len(self.all_topologies)

    def set_out_order(self, order):
        assert self.max_order >= order
        self.out_order = order
        self.active_topologies = list(range(self.in_order, self.out_order + 1))

        topology = self.topologies[self.active_topologies[-1]]
        self.out_topology = dict(lh=topology, rh=copy.deepcopy(topology))
        if isinstance(
            self.out_topology["rh"], brainnet.mesh.topology.DeepSurferTopology
        ):
            self.out_topology["rh"].reverse_face_orientation()

    def solve_ode_euler(self, h, v, dv):
        """Solve dv/dt = f(t, v) using Euler's method."""
        return v + h * dv

    # def solve_ode_RK4(self, h, v, dv):
    #     k1 = dv
    #     k2 = self.pial_deform(self.grid_sample_features(fmaps, v + h * 0.5 * k1))
    #     k3 = self.pial_deform(self.grid_sample_features(fmaps, v + h * 0.5 * k2))
    #     k4 = self.pial_deform(self.grid_sample_features(fmaps, v + h * k3))
    #     return v + h/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _get_features(self, features, maps):
        # return (torch.cat([features[m] for m in maps], dim=1), )
        return tuple(features[m] for m in maps)

    def grid_sample_features(
        self, features: list[torch.Tensor] | tuple, vertices: torch.Tensor
    ):
        return torch.cat(tuple(self.grid_sample(f, vertices) for f in features), dim=1)

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
        vertices = self.normalize_coordinates(vertices)

        # samples is N,C,D,H,W where C is from `image` and D,H,W are from `points`
        samples = torch.nn.functional.grid_sample(
            image.swapaxes(2, 4),  # N,C,W,H,D -> N,C,D,H,W
            # N,3,M -> N,M,3 -> N,D,H,W,3 where D=M; H=W=1
            vertices.mT[:, :, None, None],
            align_corners=True,
        )
        return samples[..., 0, 0]  # squeeze out H, W

    @staticmethod
    def get_image_shape(image):
        return torch.tensor(image.shape[-3:], dtype=image.dtype, device=image.device)

    def get_image_center(self, image_shape):
        return 0.5 * (image_shape[None, :, None] - 1.0)

    def set_image_center(self, image):
        """This is used with grid sampling when align_corners=True."""
        self._image_shape = self.get_image_shape(image)
        self._image_center = self.get_image_center(self._image_shape)

    def normalize_coordinates(self, coords, image: None | torch.Tensor = None):
        # vertices are in voxel coordinates

        # Transform vertices from (0, shape) to (-half_shape, half_shape), then
        # normalize to [-1, 1]
        if image is None:
            return (coords - self._image_center) / self._image_center
        else:
            center = self.get_image_center(self.get_image_shape(image))
            return (coords - center) / center

    def unnormalize_coordinates(self, coords, image: None | torch.Tensor = None):
        if image is None:
            return self._image_center * coords + self._image_center
        else:
            center = self.get_image_center(self.get_image_shape(image))
            return center * coords + center


# class SurfaceInitializerModule(GenericSurfaceModule):
#     def __init__(
#         self,
#         in_order: int,
#         out_order: int,
#         max_order: int,  # n_topologies: int = 7, # 0 - n_topologies
#         feature_maps: list[list[str]],
#         n_steps: int | list[int] | None = None,
#         topology: str = "DeepSurferTopology",
#         device: str | torch.device = "cpu",
#     ):
#         super().__init__(in_order, out_order, max_order, topology, device)

#         if n_steps is None:
#             n_steps = [2] * (self.n_topologies - 1) + [1]
#         elif isinstance(n_steps, int):
#             n_steps = [n_steps] * self.n_topologies

#         self.n_steps = dict(zip(self.all_topologies, n_steps))
#         self.step_size = {k: 1.0 / v for k, v in self.n_steps.items()}
#         self.feature_maps = dict(zip(self.all_topologies, feature_maps))

#         self.deform = torch.nn.ModuleDict()

#     def _estimate_surface(self, features: dict[str, torch.Tensor], v: torch.Tensor):
#         for order in self.active_topologies:
#             step_size = self.step_size[order]
#             deform = self.deform[str(order)]
#             fmaps = self._get_features(features, self.feature_maps[order])
#             for _ in range(self.n_steps[order]):
#                 v_features = self.grid_sample_features(fmaps, v)
#                 v = self.solve_ode_euler(step_size, v, deform(v_features))
#             if order < self.out_order:
#                 v = self.topologies[order].subdivide_vertices(v)
#         return v

#     def forward(
#         self,
#         features: dict[str, torch.Tensor],
#         template_vertices: dict[str, torch.Tensor],
#     ):
#         """
#         Faces can be retrieved from

#             faces = self.topologies[self.prediction_res].faces

#         Parameters
#         ----------
#         features : torch.Tensor
#             Tensor of shape (N, C, W, H, D) where N is batch size and C is the
#             number of channels (feature maps).
#             NOTE Torch assumes that images are (N,C,D,H,W). For convolutions
#             this does not really matter, however, when we sample features for
#             the surface vertices using `grid_sample`, we need to transpose D
#             and W to that they correspond to the coordinates of `vertices`
#             which are x,y,z (W,H,D).
#         vertices : torch.Tensor
#             Tensor of shape (N, M, 3) where M is the number of vertices and the
#             last dimension contains the coordinates (x,y,z).

#         Returns
#         -------

#         """
#         # features = [features] if isinstance(features, torch.Tensor) else features

#         # The last feature map has the same spatial dimensions as the input
#         # image
#         last_feature_map = tuple(features.values())[-1]
#         self.set_image_center(last_feature_map)

#         dtype = last_feature_map.dtype
#         template_vertices = {k: v.to(dtype) for k, v in template_vertices.items()}

#         return {
#             h: dict(white=self._estimate_surface(features, v.mT).mT)
#             for h, v in template_vertices.items()
#         }


class SurfaceModule(GenericSurfaceModule):
    def __init__(
        self,
        in_order: int,
        out_order: int,
        max_order: int,  # n_topologies: int = 7, # 0 - n_topologies
        white_feature_maps: list[list[str]],
        pial_feature_maps: list[str],
        white_n_steps: int | list[int] | None = None,
        pial_n_steps: int = 10,
        topology: str = "DeepSurferTopology",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(in_order, out_order, max_order, topology, device)

        # WHITE MATTER CONFIG
        if white_n_steps is None:
            white_n_steps = [2] * (self.n_topologies - 1) + [1]
        elif isinstance(white_n_steps, int):
            white_n_steps = [white_n_steps] * self.n_topologies

        self.white_n_steps = dict(zip(self.all_topologies, white_n_steps))
        self.white_step_size = {k: 1.0 / v for k, v in self.white_n_steps.items()}
        self.white_feature_maps = dict(zip(self.all_topologies, white_feature_maps))

        # PIAL CONFIG

        self.pial_n_steps = pial_n_steps
        self.pial_step_size = 1.0 / self.pial_n_steps
        self.pial_feature_maps = pial_feature_maps

        self.white_deform = torch.nn.ModuleDict()
        self.pial_deform = torch.nn.Module()

        self.sphere_reg = load_deepsurfer_template(self.in_order, self.device)

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
        last_feature_map = tuple(features.values())[-1]
        self.set_image_center(last_feature_map)

        dtype = last_feature_map.dtype
        template_vertices = {k: v.to(dtype) for k, v in template_vertices.items()}

        return {
            h: self._forward_hemi(h, features, v) for h, v in template_vertices.items()
        }

    def make_surface(self, hemi, vertices, vertex_data):
        s = Surface(vertices, self.out_topology[hemi])
        s.vertex_data |= vertex_data
        return s

    def _forward_hemi(
        self, hemi: str, features: dict[str, torch.Tensor], vertices: torch.Tensor
    ):
        """Predict placement of white matter surface and pial surface.."""
        white_v, white_u = self._estimate_white(features, vertices)
        white = self.make_surface(hemi, white_v, dict(sigma=white_u))
        pial_v, pial_u = self._esimate_pial(features, white.vertices)
        pial = self.make_surface(hemi, pial_v, dict(sigma=pial_u))
        return dict(white=white, pial=pial)

    def _estimate_white(self, features: dict[str, torch.Tensor], v: torch.Tensor):
        # (N, M, 3) -> (N, 3, M) such that coordinates are in the channel
        # (feature) dimension
        v = v.mT

        u = torch.zeros_like(v)
        for order in self.active_topologies:
            step_size = self.white_step_size[order]
            deform = self.white_deform[str(order)]
            fmaps = self._get_features(features, self.white_feature_maps[order])
            for _ in range(self.white_n_steps[order]):
                v_features = self.grid_sample_features(fmaps, v)

                dv, du = deform(v_features).split([3, 3], dim=1)
                # dv, du, dr = deform(v_features).split([3, 3, 3], dim=1)

                v = self.solve_ode_euler(step_size, v, dv)
                u = self.solve_ode_euler(step_size, u, du)
            if order < self.out_order:
                v = self.topologies[order].subdivide_vertices(v)
                u = self.topologies[order].subdivide_vertices(u)

        # print("white")
        # print(v.mT)
        # print(u.mT)

        # Transpose back to (N, M, 3)
        return v.mT, u.mT.exp()

    def _esimate_pial(self, features: dict[str, torch.Tensor], v: torch.Tensor):
        # (N, M, 3) -> (N, 3, M) such that coordinates are in the channel
        # (feature) dimension
        v = v.mT

        u = torch.zeros_like(v)
        fmaps = self._get_features(features, self.pial_feature_maps)
        for _ in range(self.pial_n_steps):
            v_features = self.grid_sample_features(fmaps, v)
            # dv = self.pial_deform(v_features)
            dv, du = self.pial_deform(v_features).split([3, 3], dim=1)
            # du = du.abs()

            v = self.solve_ode_euler(self.pial_step_size, v, dv)
            u = self.solve_ode_euler(self.pial_step_size, u, du)

        # print("pial")
        # print(v.mT)
        # print(u.mT)

        # Transpose back to (N, M, 3)
        return v.mT, u.mT.exp()


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
        topologies: list[brainnet.mesh.topology.Topology],
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

        in_ch = in_channels

        # Encoder
        self.encoder = torch.nn.ModuleList()
        pool_topologies = self.topologies[::-1]
        skip_channels = []
        for i, (out_ch, topo) in enumerate(
            zip(unet_channels["encoder"], pool_topologies)
        ):
            do_pool = i < max_depth - 1
            self.encoder.append(
                EncoderUnit(
                    conv_kwargs=dict(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        conv_module=conv_module,
                        topology=topo,
                        n=n_conv,
                    ),
                    pool_kwargs=dict(topology=topo, reduce=reduce),
                    do_pool=do_pool,
                )
            )
            if do_pool:
                skip_channels.append(out_ch)
            in_ch = out_ch
        skip_channels = skip_channels[::-1]

        # Decoder
        unpool_topologies = self.topologies[:-1]
        conv_topologies = self.topologies[1:]
        self.decoder = torch.nn.ModuleList()
        for out_ch, skip_ch, top_unpool, top_conv in zip(
            unet_channels["decoder"], skip_channels, unpool_topologies, conv_topologies
        ):
            self.decoder.append(
                DecoderUnit(
                    conv_kwargs=dict(
                        in_channels=in_ch + skip_ch,
                        out_channels=out_ch,
                        conv_module=conv_module,
                        topology=top_conv,
                        n=n_conv,
                    ),
                    unpool_kwargs=dict(topology=top_unpool, reduce=reduce),
                )
            )
            in_ch = out_ch
        self.out_ch = out_ch

    def get_prediction_topology(self):
        return self.topologies[-1]

    def forward(self, features):
        # Encoder
        skip_features = []
        for enc_unit in self.encoder:
            features, sf = enc_unit(features)
            if enc_unit.do_pool:
                skip_features.append(sf)

        skip_features = skip_features[::-1]

        # Decoder
        for dec_unit, sf in zip(self.decoder, skip_features):
            features = dec_unit(features, sf)

        return features


class EncoderUnit(torch.nn.Module):
    def __init__(
        self,
        conv_kwargs: dict,
        pool_kwargs: dict,
        do_pool: bool = True,
    ):
        super().__init__()

        self.do_pool = do_pool
        self.conv = torch.nn.Sequential(
            layers.ConvolutionRepeater(**conv_kwargs),
        )
        if self.do_pool:
            self.pool = layers.Pool(**pool_kwargs)

    def forward(self, features):
        features = self.conv(features)
        skip_features = features
        features = self.pool(features) if self.do_pool else features
        return features, skip_features


class Concatenate(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *args):
        return torch.concat(args, dim=self.dim)


class DecoderUnit(torch.nn.Module):
    def __init__(
        self,
        conv_kwargs: dict,
        unpool_kwargs: dict,
    ):
        super().__init__()
        self.unpool = layers.Unpool(**unpool_kwargs)
        self.concatenate = Concatenate(dim=1)
        self.conv = layers.ConvolutionRepeater(**conv_kwargs)

    def forward(self, features, skip_features):
        return self.conv(self.concatenate(self.unpool(features), skip_features))


class UNetTransform(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        topologies: list[brainnet.mesh.topology.Topology],
        channels: None | dict = None,
        unet_conv_module: torch.nn.Module = layers.EdgeConvolutionBlock,
        deform_conv_module: torch.nn.Module = layers.EdgeConvolution,
        reduction: str = "amax",
        max_depth: int = 4,
        n_convolutions: int = 1,
        deform_init_values: float | list[float] | None | tuple = 0.0,
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
        deform = deform_conv_module(
            unet.out_ch,
            out_channels,
            topologies[-1].conv_index_reduce,
            topologies[-1].conv_index_gather,
            bias=False,
            init_values=deform_init_values,
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
