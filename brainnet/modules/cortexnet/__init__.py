import torch

from brainnet.mesh import topology
from brainnet.modules.cortexnet.layers import GraphDeformationBlock

# Cortical surface reconstruction network (C)

class CortexThing(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        # topologies,
        out_res: int = 6,
        device: str | torch.device = "cpu",

        # image_shape: torch.IntTensor | torch.LongTensor,
        # config: None | config.TopoFitModelParameters = None,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)

        n_topologies = 7 # 0,1,2,3,4,5,6
        # e.g., 256 > 64 > 64 > 64 > 3
        graph_channels = [64, 64, 64]
        out_channels = 3
        n_residual_blocks = 3
        step_size_white = 1.0
        n_steps_pial = 5

        self.step_size_white = step_size_white
        self.step_size_pial = 1.0 / n_steps_pial

        assert n_topologies >= out_res + 1
        self.out_res = out_res

        # The topology is defined on the left hemisphere and although the
        # topology is the same for both hemispheres, we need to reverse the
        # order of the vertices in face array in order for the ordering to
        # remain consistent (e.g., counter-clockwise) once the vertices are
        # (almost) left-right mirrored

        # We use the left topology in the submodules which only use knowledge
        # of the neighborhoods to define the convolutions (and this is
        # independent of the face orientation).
        self.topologies = topology.get_recursively_subdivided_topology(
            n_topologies - 1,
            topology.initial_faces.to(self.device),
        )

        self.out_topology = self.topologies[out_res]

        # Surface placement modules are shared for both hemispheres

        self.white_deform = torch.nn.ModuleList()
        for topo in self.topologies:
            reduce_index, gather_index = topo.get_convolution_indices()
            self.white_deform.append(GraphDeformationBlock(
                in_channels,
                graph_channels,
                out_channels,
                reduce_index,
                gather_index,
                n_residual_blocks,
            ))

        reduce_index, gather_index = self.out_topology.get_convolution_indices()

        self.pial_deform = GraphDeformationBlock(
            in_channels,
            graph_channels,
            out_channels,
            reduce_index,
            gather_index,
            n_residual_blocks,
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
        features: torch.Tensor | list[torch.Tensor],
        initial_vertices: dict[str, torch.Tensor],
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
        # self.set_image_center(features[-1])

        return {
            h: self._forward_hemi(features, v)
            for h, v in initial_vertices.items()
        }

    def _forward_hemi(self, features, vertices):
        """Predict placement of white matter surface and pial surface.."""

        # (N, M, 3) -> (N, 3, M) such that coordinates are in the channel
        # (feature) dimension
        vertices = vertices.mT
        white_vertices = self._place_white(features, vertices)
        pial_vertices = self._place_pial(features, white_vertices)

        # NOTE Currently layer4 is not supervised so it is not valid
        # vertices_layer4 = self.linear_deform(features, vertices)
        # vertices_pial = self.linear_deform(features, vertices_layer4)

        # vertices = self.unnormalize_coordinates(vertices)
        # vertices_pial = self.unnormalize_coordinates(vertices_pial)

        # Transpose back to (N, M, 3)
        return dict(white = vertices.mT, pial = pial_vertices.mT)

    def _place_white(self, features, vertices):
        for i in range(self.out_res + 1):
            vertex_features = self.grid_sample_features(features, vertices)
            vertex_step = self.white_deform[i](vertex_features)
            vertices = vertices + self.step_size_white * vertex_step
            if i < self.out_res:
                vertices = self.topologies[i].subdivide_vertices(vertices)
        return vertices


    def _place_pial(self, features: list[torch.Tensor], vertices: torch.Tensor):
        for _ in range(self.n_steps_pial):
            vertex_features = self.grid_sample_features(features, vertices)
            vertex_step = self.pial_deform(vertex_features)
            vertices = vertices + self.step_size_pial * vertex_step
        return vertices

    def grid_sample_features(self, features: list[torch.Tensor], vertices: torch.Tensor):
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
        half_shape = (torch.as_tensor(image.shape[-3:], device=image.device) - 1) / 2
        points = (vertices.mT - half_shape) / half_shape # N,3,M -> N,M,3

        # points = vertices.mT

        # samples is N,C,D,H,W where C is from `image` and D,H,W are from `points`
        samples = torch.nn.functional.grid_sample(
            image.swapaxes(2,4),        # N,C,W,H,D -> N,C,D,H,W
            points[:, :, None, None],   # N,M,3     -> N,D,H,W,3 where D=M; H=W=1
            align_corners=True
        )
        return samples[..., 0, 0] # squeeze out H, W


    # def set_image_center(self, image):
    #     """This is used with grid sampling with align_corners=True."""
    #     self._image_shape = torch.tensor(image.shape[2:], device=image.device)
    #     center = 0.5 * (self._image_shape - 1.0)
    #     self._image_center = center[None, :, None]

    # def normalize_coordinates(self, coords):
    #     # vertices are in voxel coordinates

    #     # Transform vertices from (0, shape) to (-half_shape, half_shape), then
    #     # normalize to [-1, 1]
    #     return (coords - self._image_center) / self._image_center
    #     # return torch.clip((v - self._image_center) / self._image_center, -1.0, 1.0)

    # def unnormalize_coordinates(self, coords):
    #     return self._image_center * coords + self._image_center
