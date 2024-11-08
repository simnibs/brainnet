import torch

from brainnet.modules.graph import layers
from brainnet.modules.graph.modules import SurfaceModule

# Cortical surface reconstruction network (C)


white_kwargs = dict(
    graph_channels=[64, 64, 64],
    n_residual_blocks = 1,
    n_steps = [2, 2, 2, 2, 2, 2, 1],
    feaure_maps = [
        ["encoder:3", "decoder:0"],
        ["encoder:2", "decoder:1"],
        ["encoder:1", "decoder:2"],
        ["encoder:0", "decoder:3"],
        ["encoder:0", "decoder:3"],
        ["encoder:0", "decoder:3"],
        ["encoder:0", "decoder:3"],
    ],
)
pial_kwargs = dict(
    graph_channels=[64],
    n_steps = 10,
    feaure_maps = ["encoder:0", "decoder:3"],
)

white_n_steps = [2, 2, 2, 2, 2, 2, 1]
white_feature_maps = [
    ["encoder:3", "decoder:0"],
    ["encoder:2", "decoder:1"],
    ["encoder:1", "decoder:2"],
    ["encoder:0", "decoder:3"],
    ["encoder:0", "decoder:3"],
    ["encoder:0", "decoder:3"],
    ["encoder:0", "decoder:3"],
]

class CortexThing(SurfaceModule):
    def __init__(
        self,
        in_channels: dict[str, int],  # int,
        # topologies,
        out_res: int = 6,
        white_kwargs: dict | None = None,
        pial_kwargs: dict | None = None,
        device: str | torch.device = "cpu",
        # image_shape: torch.IntTensor | torch.LongTensor,
        # config: None | config.TopoFitModelParameters = None,
    ) -> None:
        super().__init__(out_res, white_kwargs, pial_kwargs, device)

        out_channels = 3

        white_graph_channels = [64, 64, 64]
        white_n_residual_blocks = 1

        pial_graph_channels = [64]


        # Surface placement modules are shared for both hemispheres

        self.white_deform = torch.nn.ModuleList()
        for topo, p in zip(self.topologies, self.white_feature_maps):
            reduce_index, gather_index = topo.get_convolution_indices()
            self.white_deform.append(
                # layers.GraphConvolutionDeformationBlock(
                #     sum(in_channels[i] for i in p),
                #     white_graph_channels,
                #     out_channels,
                #     reduce_index,
                #     gather_index,
                # )
                layers.ResidualGraphConvolutionDeformationBlock(
                    sum(in_channels[i] for i in p),
                    white_graph_channels,
                    out_channels,
                    reduce_index,
                    gather_index,
                    white_n_residual_blocks,
                )
            )

        reduce_index, gather_index = self.out_topology.get_convolution_indices()

        self.pial_deform = layers.GraphConvolutionDeformationBlock(
            sum(in_channels[i] for i in self.pial_feature_maps),
            pial_graph_channels,
            out_channels,
            reduce_index,
            gather_index,
        )

        # layer 4 and pial placement
        # NOTE one or two linear deformation layers? shared or different parameters?
        # self.linear_deform = layers.GraphLinearDeform(
        #     in_channels,
        #     config.linear_deform.channels,
        #     config.linear_deform.n_iterations,
        #     topology=self.get_prediction_topology(),
        # )

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
        # vertices = self.normalize_coordinates(vertices)

        white_vertices = self._place_white(features, vertices)
        pial_vertices = self._place_pial(features, white_vertices)

        # white_vertices = self.unnormalize_coordinates(white_vertices).mT
        # pial_vertices = self.unnormalize_coordinates(pial_vertices).mT
        white_vertices = white_vertices.mT
        pial_vertices = pial_vertices.mT

        # NOTE Currently layer4 is not supervised so it is not valid
        # vertices_layer4 = self.linear_deform(features, vertices)
        # vertices_pial = self.linear_deform(features, vertices_layer4)

        # Transpose back to (N, M, 3)
        return dict(white=white_vertices, pial=pial_vertices)

    def _get_features(self, features, maps):
        return torch.cat([features[m] for m in maps], dim=1)

    def _place_white(self, features: dict[str, torch.Tensor], v: torch.Tensor):
        for i in range(self.out_res + 1):
            step_size = self.white_step_size[i]
            deform = self.white_deform[i]
            fmap = self._get_features(features, self.white_feature_maps[i])
            for _ in range(self.white_n_steps[i]):
                v_features = self.grid_sample(fmap, v)
                v = v + step_size * deform(v_features)
            if i < self.out_res:
                v = self.topologies[i].subdivide_vertices(v)
        return v

    def _place_pial(self, features: dict[str, torch.Tensor], v: torch.Tensor):
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
            vertices.mT[
                :, :, None, None
            ],  # N,3,M -> N,M,3 -> N,D,H,W,3 where D=M; H=W=1
            align_corners=True,
        )
        return samples[..., 0, 0]  # squeeze out H, W

    def set_image_center(self, image):
        """This is used with grid sampling with align_corners=True."""
        self._image_shape = torch.tensor(image.shape[-3:], device=image.device)
        center = 0.5 * (self._image_shape - 1.0)
        self._image_center = center[None, :, None]

    def normalize_coordinates(self, coords):
        # vertices are in voxel coordinates

        # Transform vertices from (0, shape) to (-half_shape, half_shape), then
        # normalize to [-1, 1]
        return (coords - self._image_center) / self._image_center
        # return torch.clip((v - self._image_center) / self._image_center, -1.0, 1.0)

    def unnormalize_coordinates(self, coords):
        return self._image_center * coords + self._image_center




# class SphericalReg(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: dict[str, int],  # int,
#         # topologies,
#         device: str | torch.device = "cpu",
#         # image_shape: torch.IntTensor | torch.LongTensor,
#         # config: None | config.TopoFitModelParameters = None,
#     ) -> None:
#         super().__init__()

#         self.device = torch.device(device)

#         # CONFIGURATION

#         n_topologies = 7  # 0,1,2,3,4,5,6

#         # e.g., # image features > 64 > 64 > 3
#         # white_graph_channels = [32, 32]
#         white_graph_channels = [64, 64, 64]
#         white_n_residual_blocks = 1
#         white_n_steps = [2, 2, 2, 2, 2, 2, 1]
#         white_feature_maps = [
#             ["encoder:3", "decoder:0"],
#             ["encoder:2", "decoder:1"],
#             ["encoder:1", "decoder:2"],
#             ["encoder:0", "decoder:3"],
#             ["encoder:0", "decoder:3"],
#             ["encoder:0", "decoder:3"],
#             ["encoder:0", "decoder:3"],
#         ]
#         # white_feature_maps = 7 * ["decoder:3"]

#         # image features > 64 > 3
#         # pial_graph_channels = [32]
#         pial_graph_channels = [64]
#         # pial_n_residual_blocks = 0
#         pial_n_steps = 10
#         pial_feature_maps = ["encoder:0", "decoder:3"]

#         s2s_out_channels = 3

#         ####

#         self.white_n_steps = white_n_steps
#         self.white_step_size = [1.0 / i for i in white_n_steps]
#         self.white_feature_maps = white_feature_maps

#         self.pial_n_steps = pial_n_steps
#         self.pial_step_size = 1.0 / pial_n_steps
#         self.pial_feature_maps = pial_feature_maps

#         assert n_topologies >= out_res + 1
#         self.out_res = out_res

#         # The topology is defined on the left hemisphere and although the
#         # topology is the same for both hemispheres, we need to reverse the
#         # order of the vertices in face array in order for the ordering to
#         # remain consistent (e.g., counter-clockwise) once the vertices are
#         # (almost) left-right mirrored

#         # We use the left topology in the submodules which only use knowledge
#         # of the neighborhoods to define the convolutions (and this is
#         # independent of the face orientation).
#         self.topologies = topology.get_recursively_subdivided_topology(
#             n_topologies - 1,
#             topology.initial_faces.to(self.device),
#         )

#         self.out_topology = self.topologies[out_res]

#         assert len(self.topologies) == len(self.white_n_steps)

#         # Surface placement modules are shared for both hemispheres

#         s2s_graph_channels = [64, 64, 64]
#         s2s_in_channels =
#         s2s_out_channels = 2
#         topo = self.topologies[0]
#         reduce_index, gather_index = topo.get_convolution_indices()
#         self.surface_to_sphere = layers.GraphConvolutionDeformationBlock(
#             in_channels,
#             s2s_graph_channels,
#             s2s_out_channels,
#             reduce_index,
#             gather_index,
#         )

#         self.white_deform = torch.nn.ModuleList()
#         for topo, p in zip(self.topologies, self.white_feature_maps):
#             reduce_index, gather_index = topo.get_convolution_indices()
#             self.white_deform.append(
#                 # layers.GraphConvolutionDeformationBlock(
#                 #     sum(in_channels[i] for i in p),
#                 #     white_graph_channels,
#                 #     out_channels,
#                 #     reduce_index,
#                 #     gather_index,
#                 # )
#                 layers.ResidualGraphConvolutionDeformationBlock(
#                     sum(in_channels[i] for i in p),
#                     white_graph_channels,
#                     s2s_out_channels,
#                     reduce_index,
#                     gather_index,
#                     white_n_residual_blocks,
#                 )
#             )

#         reduce_index, gather_index = self.out_topology.get_convolution_indices()

#         # self.pial_deform = layers.GraphConvolutionDeformationBlock(
#         #     sum(in_channels[i] for i in self.pial_feature_maps),
#         #     pial_graph_channels,
#         #     out_channels,
#         #     reduce_index,
#         #     gather_index,
#         # )

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
#         self.set_image_center(features["decoder:3"])

#         return {
#             h: self._forward_hemi(features, v) for h, v in template_vertices.items()
#         }

#     def _forward_hemi(self, features: dict[str, torch.Tensor], vertices: torch.Tensor):
#         """Predict placement of white matter surface and pial surface.."""

#         # (N, M, 3) -> (N, 3, M) such that coordinates are in the channel
#         # (feature) dimension and normalize
#         vertices = vertices.mT

#         spherical_vertices = self.initialize(features, vertices)
#         spherical_vertices = self.register(features, vertices)
#         # Transpose back to (N, M, 3)
#         spherical_vertices = spherical_vertices.mT
#         return spherical_vertices

#     def _get_features(self, features, maps):
#         return torch.cat([features[m] for m in maps], dim=1)

#     def initialize(self, features, vertices):
#         res = 0
#         v = vertices[:, :self.topologies[res].n_vertices]

#         fmap = self._get_features(features, self.white_feature_maps[res])

#         v_features = self.grid_sample(fmap, v)

#         v_sphere = self.


#     def register(self, features: dict[str, torch.Tensor], v: torch.Tensor):
#         for i in range(self.out_res + 1):
#             step_size = self.white_step_size[i]
#             deform = self.white_deform[i]
#             fmap = self._get_features(features, self.white_feature_maps[i])
#             for _ in range(self.white_n_steps[i]):
#                 v_features = self.grid_sample(fmap, v)
#                 v = v + step_size * deform(v_features)
#             if i < self.out_res:
#                 v = self.topologies[i].subdivide_vertices(v)
#         return v

#     def _place_pial(self, features: dict[str, torch.Tensor], v: torch.Tensor):
#         fmap = self._get_features(features, self.pial_feature_maps)
#         for _ in range(self.pial_n_steps):
#             v_features = self.grid_sample(fmap, v)
#             v = v + self.pial_step_size * self.pial_deform(v_features)
#         return v

#     def grid_sample_features(
#         self, features: list[torch.Tensor], vertices: torch.Tensor
#     ):
#         return torch.cat([self.grid_sample(f, vertices) for f in features], dim=1)

#     def grid_sample(self, image, vertices):
#         """


#         Parameters
#         ----------
#         image :
#             image shape is (N, C, W, H, D)
#         vertices :
#             vertices shape is (N, 3, M)

#         Returns
#         -------
#         samples :
#             samples shape (N, C, M)
#         """

#         # vertices are in voxel coordinates

#         # Transform vertices from (0, shape) to (-half_shape, half_shape), then
#         # normalize to [-1, 1]
#         # half_shape = (torch.as_tensor(image.shape[-3:], device=image.device) - 1) / 2
#         # points = (vertices.mT - half_shape) / half_shape # N,3,M -> N,M,3

#         # points = vertices.mT
#         vertices = self.normalize_coordinates(vertices)

#         # samples is N,C,D,H,W where C is from `image` and D,H,W are from `points`
#         samples = torch.nn.functional.grid_sample(
#             image.swapaxes(2, 4),  # N,C,W,H,D -> N,C,D,H,W
#             vertices.mT[
#                 :, :, None, None
#             ],  # N,3,M -> N,M,3 -> N,D,H,W,3 where D=M; H=W=1
#             align_corners=True,
#         )
#         return samples[..., 0, 0]  # squeeze out H, W

#     def set_image_center(self, image):
#         """This is used with grid sampling with align_corners=True."""
#         self._image_shape = torch.tensor(image.shape[-3:], device=image.device)
#         center = 0.5 * (self._image_shape - 1.0)
#         self._image_center = center[None, :, None]

#     def normalize_coordinates(self, coords):
#         # vertices are in voxel coordinates

#         # Transform vertices from (0, shape) to (-half_shape, half_shape), then
#         # normalize to [-1, 1]
#         return (coords - self._image_center) / self._image_center
#         # return torch.clip((v - self._image_center) / self._image_center, -1.0, 1.0)

#     def unnormalize_coordinates(self, coords):
#         return self._image_center * coords + self._image_center

