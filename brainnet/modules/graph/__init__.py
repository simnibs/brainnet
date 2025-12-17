import copy
from typing import Type

import torch

import brainnet.mesh.topology
from brainnet.mesh.surface import load_deepsurfer_template, rotate, Surface
from .layers import EdgeConvolution, EdgeConvolutionBlock
from .unet import UNet


class UNetDeformBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        topologies: list[brainnet.mesh.topology.Topology],
        channels: None | dict = None,
        unet_conv_module: Type[torch.nn.Module] = EdgeConvolutionBlock,
        deform_conv_module: Type[torch.nn.Module] = EdgeConvolution,
        reduction: str = "amax",
        max_depth: int = 4,
        n_convolutions: int = 1,
        deform_init_values: float | list[float] | None | tuple = 0.0,
    ) -> None:
        """This module uses a UNet defined on the graph to extract features
        and subsequently transforms these to deformation vectors using.

            features = unet_module(features)
            dV = deformation_module(features)

        """
        super().__init__()

        # # Initialize this way to get parameter._version == 1 like all the rest
        # self.n_steps = n_steps
        # self.step_size = torch.nn.Parameter(torch.empty([1]))
        # torch.nn.init.constant_(self.step_size, 1.0 / n_steps)

        self.out_groups = out_channels

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
        self.append(unet)
        self.append(deform)


class GenericSurfaceModule(torch.nn.Module):
    def __init__(
        self,
        in_order: int,
        out_order: int,
        max_order: int,  # n_topologies: int = 7, # 0 - n_topologies
        topology: str = "DeepSurferTopology",
    ):
        super().__init__()
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
        ).recursive_subdivision(self.max_order)
        self.all_topologies = list(range(self.in_order, self.max_order + 1))
        self.n_topologies = len(self.all_topologies)

    def set_out_order(self, order):
        assert self.max_order >= order
        self.out_order = order
        self.active_topologies = list(range(self.in_order, self.out_order + 1))

        topology = self.topologies[self.active_topologies[-1]]
        self.out_topology = torch.nn.ModuleDict(
            dict(lh=topology, rh=copy.deepcopy(topology))
        )
        if isinstance(
            self.out_topology["rh"], brainnet.mesh.topology.DeepSurferTopology
        ):
            self.out_topology["rh"].reverse_face_orientation()

    def solve_ode_euler(self, h, v, dv):
        """Solve dv/dt = f(t, v) using Euler's method."""
        return v + h * dv

    def solve_ode_euler_rotate(self, h, v, dv):
        """Rotate v about the (normalized) vector dv by an angle of h * |dv|."""
        v = v.mT
        dv = dv.mT
        theta = torch.linalg.vector_norm(dv, dim=2, keepdim=True, dtype=dv.dtype)
        k = dv / theta
        return rotate(v, k, h * theta, normalize=False).mT

    def _project_to_sphere(self, r, dim: int = 1):
        return (
            r
            * self.sphere_reg_radius
            / torch.linalg.vector_norm(r, dim=dim, keepdim=True, dtype=r.dtype)
        )

    # def solve_ode_RK4(self, h, v, dv):
    #     k1 = dv
    #     k2 = self.pial_deform(self.grid_sample_features(fmaps, v + h * 0.5 * k1))
    #     k3 = self.pial_deform(self.grid_sample_features(fmaps, v + h * 0.5 * k2))
    #     k4 = self.pial_deform(self.grid_sample_features(fmaps, v + h * k3))
    #     return v + h/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _get_features(self, features, maps):
        # return (torch.cat([features[m] for m in maps], dim=1), )
        return tuple(features[m] for m in maps)

        # with torch.autocast("cuda", torch.float32, enabled=True):

    def grid_sample_features(
        self, features: list[torch.Tensor] | tuple, vertices: torch.Tensor
    ):
        return torch.cat(tuple(self.grid_sample(f, vertices) for f in features), dim=1)

    # @torch.autocast("cuda", torch.float32, enabled=True)
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
    ) -> None:
        super().__init__(in_order, out_order, max_order, topology)

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

        self.sphere_reg_radius = 100.0  # Output radius
        self.sphere_reg = load_deepsurfer_template(self.in_order, "sphere.reg")
        for k, v in self.sphere_reg.items():
            self.sphere_reg[k].vertices /= torch.linalg.vector_norm(
                v.vertices, dim=2, keepdim=True
            )
            self.sphere_reg[k].vertices *= self.sphere_reg_radius

    def forward(
        self, features: dict[str, torch.Tensor], template: dict[str, torch.Tensor]
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
        template = {k: v.to(dtype) for k, v in template.items()}

        return {h: self._forward_hemi(h, features, v) for h, v in template.items()}

    def make_surface(self, hemi, vertices, vertex_data: dict | None = None):
        s = Surface(vertices, self.out_topology[hemi])
        if vertex_data is not None:
            s.vertex_data |= vertex_data
        return s

    def _forward_hemi(
        self, hemi: str, features: dict[str, torch.Tensor], vertices: torch.Tensor
    ):
        """Predict placement of white matter surface and pial surface."""
        reg = self.sphere_reg[hemi].vertices.detach().clone()
        reg = reg.to(vertices.dtype)

        wv, wu, wr = self._estimate_white(features, vertices, reg)
        pv, pu = self._esimate_pial(features, wv, wu)

        wu = wu.exp()  # log(wu) -> wu
        pu = pu.exp()

        return {
            "white": self.make_surface(hemi, wv, dict(sigma=wu)),
            "pial": self.make_surface(hemi, pv, dict(sigma=pu)),
            "sphere.reg": self.make_surface(hemi, wr),
        }

    def _estimate_white(
        self, features: dict[str, torch.Tensor], v: torch.Tensor, r: torch.Tensor
    ):
        # (N, M, 3) -> (N, 3, M) such that coordinates are in the channel
        # (feature) dimension
        v = v.mT
        u = torch.zeros_like(v)
        r = r.mT
        for order in self.active_topologies:
            step_size = self.white_step_size[order]
            deform = self.white_deform[str(order)]
            fmaps = self._get_features(features, self.white_feature_maps[order])
            for _ in range(self.white_n_steps[order]):
                v_features = self.grid_sample_features(fmaps, v)

                dv, du, dr = deform(v_features).split(self.white_out_groups, dim=1)
                v = self.solve_ode_euler(step_size, v, dv)
                u = self.solve_ode_euler(step_size, u, du)
                # r = self.solve_ode_euler_rotate(step_size, r, dr)
                r = self.solve_ode_euler(step_size, r, dr)
                r = self._project_to_sphere(r)
            if order < self.out_order:
                v = self.topologies[order].subdivide_vertices(v.mT).mT
                u = self.topologies[order].subdivide_vertices(u.mT).mT
                r = self.topologies[order].subdivide_vertices(r.mT).mT
                r = self._project_to_sphere(r)

        # Transpose back to (N, M, 3)
        return v.mT, u.mT, r.mT

    def _esimate_pial(
        self,
        features: dict[str, torch.Tensor],
        v: torch.Tensor,
        init_sigma: torch.Tensor | None = None,
    ):
        # (N, M, 3) -> (N, 3, M) such that coordinates are in the channel
        # (feature) dimension
        v = v.mT

        u = torch.zeros_like(v) if init_sigma is None else init_sigma.mT
        fmaps = self._get_features(features, self.pial_feature_maps)
        for _ in range(self.pial_n_steps):
            v_features = self.grid_sample_features(fmaps, v)

            dv, du = self.pial_deform(v_features).split(self.pial_out_groups, dim=1)

            v = self.solve_ode_euler(self.pial_step_size, v, dv)
            u = self.solve_ode_euler(self.pial_step_size, u, du)

        # Transpose back to (N, M, 3)
        return v.mT, u.mT


class TopoFit(SurfaceModule):
    def __init__(
        self,
        in_channels: dict[str, int],
        out_channels: int = 3,
        white_channels: dict | None = None,
        pial_channels: list | None = None,
        pial_deform_module: str = "LinearDeformationBlock",
        predict_uncertainty: bool = True,
        predict_registration: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """_summary_

        Parameters
        ----------
        in_channels : dict[str, int]
            _description_
        out_channels : int, optional
            _description_, by default 9
        white_channels : dict | None, optional
            _description_, by default None
        pial_channels : list | None, optional
            _description_, by default None
        pial_deform_module : str, optional
            _description_, by default "LinearDeformationBlock"

        Raises
        ------
        ValueError
            _description_
        """
        super().__init__(*args, **kwargs)

        # Define output channels and initial values of the last layer
        self.white_out_groups = [out_channels]
        white_deform_init = out_channels * [0.0]
        if predict_uncertainty:
            self.white_out_groups.append(out_channels)
            white_deform_init += out_channels * [0.0]
        if predict_registration:
            self.white_out_groups.append(out_channels)
            white_deform_init += out_channels * [0.0]

        self.pial_out_groups = [out_channels]
        pial_deform_init = out_channels * [0.01]
        if predict_uncertainty:
            self.pial_out_groups.append(out_channels)
            pial_deform_init += out_channels * [0.0]

        if white_channels is None:
            white_channels = dict(encoder=[64, 64, 64, 64], decoder=[64, 64, 64])
        pial_channels = [32] if pial_channels is None else pial_channels

        UNetTransform_kwargs = dict(
            channels=white_channels,
            deform_init_values=white_deform_init,
        )

        for topo in self.all_topologies:  # e.g., 1, 2, ..., 7
            self.white_deform[str(topo)] = UNetDeformBlock(
                sum(in_channels[j] for j in self.white_feature_maps[topo]),
                sum(self.white_out_groups),
                self.topologies[: topo + 1],
                **UNetTransform_kwargs,
            )

        m = getattr(brainnet.modules.graph.layers, pial_deform_module)
        match pial_deform_module:
            case "LinearDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    sum(self.pial_out_groups),
                    out_init_values=pial_deform_init,
                )
            case "GraphConvolutionDeformationBlock" | "EdgeConvolutionDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    sum(self.pial_out_groups),
                    self.out_topology.conv_index_reduce,
                    self.out_topology.conv_index_gather,
                    out_init_values=pial_deform_init,
                )
            case "ResidualGraphConvolutionDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    sum(self.pial_out_groups),
                    self.out_topology.conv_index_reduce,
                    self.out_topology.conv_index_gather,
                    n_residual_blocks=3,
                    out_init_values=pial_deform_init,
                )
            case _:
                raise ValueError(
                    f"Invalid module for pial deformation ({pial_deform_module})"
                )


# class TopoReg(GenericSurfaceModule):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int = 3,
#         in_order: int = 0,
#         out_order: int = 6,
#         max_order: int = 6,
#         topology: str = "DeepSurferTopology",
#         n_steps: int | list[int] | None = None,
#     ):
#         super().__init__(in_order, out_order, max_order, topology)

#         if n_steps is None:
#             n_steps = [2] * (self.n_topologies - 1) + [1]
#         elif isinstance(n_steps, int):
#             n_steps = [n_steps] * self.n_topologies

#         self.n_steps = dict(zip(self.all_topologies, n_steps))
#         self.step_size = {k: 1.0 / v for k, v in self.n_steps.items()}


#         feature_extractor_channels = dict(encoder=[64, 64, 64, 64], decoder=[64, 64, 64])
#         registration_channels = dict(encoder=[64, 64, 64, 64], decoder=[64, 64, 64])

#         # surface feature extractor
#         self.unet = UNet(
#             in_channels,
#             self.topologies,
#             EdgeConvolutionBlock,
#             reduce="amax",
#             channels=feature_extractor_channels,
#             max_depth=4,
#             n_conv=1,
#         )
#         out_ch = feature_extractor_channels["decoder"][-1]


#         # surface registration
#         self.deform = torch.nn.ModuleDict()
#         for topo in self.all_topologies:  # e.g., 1, 2, ..., 7
#             self.deform[str(topo)] = UNetDeformBlock(
#                 2 * out_ch,
#                 out_channels,
#                 self.topologies[: topo + 1],
#                 registration_channels,
#                 deform_init_values=deform_init,
#             )


#         self.sphere_reg_radius = 100.0  # Output radius
#         self.sphere_reg = load_deepsurfer_template(self.max_order, "sphere.reg")
#         for k, v in self.sphere_reg.items():
#             self.sphere_reg[k].vertices /= torch.linalg.vector_norm(
#                 v.vertices, dim=2, keepdim=True
#             )
#             self.sphere_reg[k].vertices *= self.sphere_reg_radius


#         self.white = load_deepsurfer_template(self.max_order, "white")

#         self.interpolater =


#     def to_gridded_data(self, order, f):
#         return f[]


#     def grid_sample_features(
#         self, features: list[torch.Tensor] | tuple, vertices: torch.Tensor
#     ):
#         return torch.cat(tuple(self.grid_sample(f, vertices) for f in features), dim=1)

#     # @torch.autocast("cuda", torch.float32, enabled=True)
#     def grid_sample(self, image, vertices):
#         """

#         Parameters
#         ----------
#         image :
#             image shape is (N, C, W, H)
#         vertices :
#             vertices shape is (N, 2, M)

#         Returns
#         -------
#         samples :
#             samples shape (N, C, M)
#         """
#         # vertices are in voxel coordinates
#         vertices = self.normalize_coordinates(vertices)

#         # samples is N,C,D,H,W where C is from `image` and D,H,W are from `points`
#         samples = torch.nn.functional.grid_sample(
#             image, # N,C,H,W
#             # N,2,M -> N,M,2 -> N,H,W,2 where H=M; W=1
#             vertices.mT[:, :, None],
#             align_corners=True,
#         )
#         return samples[..., 0, 0]  # squeeze out H, W


#     def forward(self, vertices: dict[str,torch.Tensor]):

#         out = {h: self._forward_hemi(v, self.white[h].vertices) for h, v in vertices.items()}


#     def _forward_hemi(self, v0: torch.Tensor, v1: torch.Tensor):


#         for order in self.active_topologies:
#             # (N, M, 3) -> (N, 3, M) such that coordinates are in the channel
#             # (feature) dimension
#             nv = self.topologies[order].n_vertices
#             f0 = self.unet(v0[:, :v].mT) # Moving surface
#             f1 = self.unet(v1[:, :v].mT) # Fixed surface
#             f1 = self.to_gridded_data(order, f1) # Interpolate to 2D grid

#             step_size = self.step_size[order]
#             deform = self.deform[str(order)]
#             fmaps = self._get_features(features, self.white_feature_maps[order])
#             for _ in range(self.n_steps[order]):
#                 v_features = self.grid_sample_features(fmaps, v)

#                 dv, du, dr = deform(v_features).split(self.white_out_groups, dim=1)
#                 v = self.solve_ode_euler(step_size, v, dv)
#                 u = self.solve_ode_euler(step_size, u, du)
#                 # r = self.solve_ode_euler_rotate(step_size, r, dr)
#                 r = self.solve_ode_euler(step_size, r, dr)
#                 r = self._project_to_sphere(r)
#             if order < self.out_order:
#                 v = self.topologies[order].subdivide_vertices(v.mT).mT
#                 u = self.topologies[order].subdivide_vertices(u.mT).mT
#                 r = self.topologies[order].subdivide_vertices(r.mT).mT
#                 r = self._project_to_sphere(r)

#         # Transpose back to (N, M, 3)
#         return v.mT, u.mT.exp(), r.mT
