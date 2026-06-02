import copy
import torch

import brainsynth
from brainsynth.utilities import apply_affine
from brainsynth.transforms.spatial import FlipSurface
import brainnet.mesh.topology
from brainnet.utils import atleast_nd_append, atleast_nd_prepend

try:
    from brainnet.mesh.cuda import extensions as cuda_extensions
except ImportError:
    pass

"""
topology = rst[-1]
curv = torch.rand((topology.n_vertices,3))
reduce_index, gather_index = topology.vertex_adjacency.T
smooth_curv = torch.zeros_like(curv); smooth_curv.index_add_(0, reduce_index, curv[gather_index])
"""

# def compute_self_intersections(vertices: torch.Tensor, faces: torch.IntTensor):
#     # assert self.vertices.dtype == torch.float
#     # assert self.get_faces().dtype == torch.int
#     # vertices = self.vertices.detach()
#     # faces = self.get_faces().detach()

#     # the extension returns (intersecting triangles, # intersecting triangles)
#     return cuda_extensions.compute_self_intersections(vertices, faces)


# class ComputeSelfIntersections(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, vertices, faces):
#         print(vertices.requires_grad, faces.requires_grad)
#         ctx.save_for_backward(vertices)
#         ctx.faces = faces
#         print(ctx.saved_tensors)
#         # `compute_self_intersections` returns a tuple of two outputs
#         # - a vector of len(faces) with 1 (0) for triangles which has (has no)
#         # intersections
#         # - total number of intersecting triangles
#         return cuda_extensions.compute_self_intersections(vertices, faces)

#     @staticmethod
#     def backward(ctx, grad_output):
#         print(ctx.saved_tensors)
#         print(grad_output)
#         (vertices,) = ctx.saved_tensors
#         faces = ctx.faces
#         intersect, _ = grad_output
#         grad_vertices = None

#         if ctx.needs_input_grad[0]:
#             # for a vertex, we estimate the gradient as the number of faces
#             # which intersect
#             grad_vertices = torch.broadcast_to(
#                 torch.bincount(
#                     faces[intersect.to(bool)].ravel(), minlength=len(vertices)
#                 )[:, None],
#                 vertices.shape,
#             ).float()
#         # if ctx.needs_input_grad[1]:
#         #     grad_faces = intersect
#         return grad_vertices


def _is_vector_data(d):
    return d.ndim == 3 and d.shape[-1] == 3


def rotate(v, k: torch.Tensor, theta: torch.Tensor, normalize: bool = True):
    """Rotate `v` by `theta` (angle) around `k` (axis).

    Rodrigues' rotation formula.

    Parameters
    ----------
    v : torch.Tensor
        Points to rotate
    k : torch.Tensor
        Axis around which to rotate. Either one axis for all vertices
        (k.shape = (3,)) or one axis per vertex (k.shape = (..., 3)).
    alpha : torch.Tensor
    normalize : bool


    References
    ----------
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    """

    k = torch.nn.functional.normalize(k, dim=-1) if normalize else k
    k_like_v = k.expand_as(v)

    # components of v that are parallel and perpendicular to k
    # (only the perpendicular component is affected by the rotation)
    v_para = torch.sum(k_like_v * v, dim=-1, keepdim=True, dtype=v.dtype) * k_like_v
    v_perp = v - v_para
    # coordinates in the plane (whose normal is k) spanned by v_perp and k x v
    v_perp_x = theta.cos() * v_perp
    v_perp_y = torch.linalg.cross(k_like_v, v) * theta.sin()
    return v_para + v_perp_x + v_perp_y


class InterpolatedData(torch.nn.Module):
    def __init__(
        self,
        points: torch.Tensor | None = None,
        face_index: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        data: dict[str, torch.Tensor] | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = device

        self.points = points
        self.face_index = face_index
        self.weights = weights

        self.data = data

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        if value is None:
            value = torch.tensor([], device=self.device)
        self.register_buffer("_points", value, persistent=False)

    @property
    def face_index(self):
        return self._face_index

    @face_index.setter
    def face_index(self, value):
        if value is None:
            value = torch.tensor([], dtype=torch.int, device=self.device)
        self.register_buffer("_face_index", value, persistent=False)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if value is None:
            value = torch.tensor([], device=self.device)
        self.register_buffer("_weights", value, persistent=False)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if value is None:
            self._data = {}
        else:
            self._data


class Surface(torch.nn.Module):
    def __init__(
        self,
        vertices: torch.Tensor,
        topology: brainnet.mesh.topology.Topology | torch.Tensor,
        topology_class: str = "DeepSurferTopology",
        vertex_data=None,
        face_data=None,
        interpolated_data=None,
    ) -> None:
        """A batch of surfaces (vertices) that share a topology.

        Parameters
        ----------
        vertices : torch.Tensor
            _description_
        topology : Topology | torch.Tensor
            If a tensor, then it is assumed to be a 2d array representing the
            connectivity of the surface.
        """
        super().__init__()

        self._topology_class = getattr(brainnet.mesh.topology, topology_class)
        self.topology = topology
        self.vertices = vertices

        self.vertex_data = {} if vertex_data is None else vertex_data
        self.face_data = {} if face_data is None else face_data
        if interpolated_data is None:
            self.interpolated = InterpolatedData(device=vertices.device)
        else:
            self.interpolated = interpolated_data

    @property
    def topology(self):
        return self._topology

    @topology.setter
    def topology(self, value):
        self._topology = (
            value
            if issubclass(value.__class__, brainnet.mesh.topology.Topology)
            else self._topology_class(value)
        )

    def get_faces(self):
        return self.topology.faces

    @property
    def vertices(self) -> torch.Tensor:
        return self._vertices

    @vertices.setter
    def vertices(self, value: torch.Tensor):
        value = atleast_nd_prepend(value, 3)
        assert (
            value.shape[1] == self.topology.n_vertices
        ), f"Vertices dimension mismatch: {value.shape[1]} and {self.topology.n_vertices}"
        self.n_batch, _, self.n_dim = value.shape

        # batch indexer (e.g., [[0, 1, 2]])
        self.register_buffer("_vertices", value, persistent=False)
        self.register_buffer(
            "batch_ix",
            torch.arange(self.n_batch, device=value.device),
            persistent=False,
        )

    def get_device(self):
        return self.vertices.device

    # @property
    # def mean_curvature_vector(self):
    #     return self._mean_curvature_vector

    # @mean_curvature_vector.setter
    # def mean_curvature_vector(self, value):
    #     value = torch.atleast_3d(value)
    #     assert value.shape[1] == self.topology.n_vertices
    #     self._mean_curvature_vector = value

    def as_mesh(self):
        # (n_batch, n_vertices, v_per_face, coordinates)
        return self.vertices[:, self.get_faces()]

    def bounding_box(self):
        """(batch, 2, 3)."""
        return torch.stack((self.vertices.amin(1), self.vertices.amax(1)), dim=1)

    def center_on_origin(self):
        center = self.bounding_box().mean(1)[:, None]
        self.vertices = self.vertices - center

    def compute_face_barycenters(self):
        return self.as_mesh().mean(2)

    def flip(self, dim: int, size: torch.Size | None = None, inplace: bool = False):
        flip = FlipSurface(dim, size)

        out = (
            self
            if inplace
            else Surface(
                self.vertices,
                copy.deepcopy(self.topology),
                vertex_data=self.vertex_data,
                face_data=self.face_data,
                interpolated_data=self.interpolated,
            )
        )
        out.vertices = flip(out.vertices)
        out.topology.reverse_face_orientation()

        # Flip vector data only
        # for k, v in out.vertex_data.items():
        #     if _is_vector_data(v):
        #         print(f"Flipping vertex data {k}")
        #         print(v[:, :5])
        #         out.vertex_data[k] = flip(v)
        #         print(out.vertex_data[k][:, :5])

        # for k, v in out.face_data.items():
        #     if _is_vector_data(v):
        #         print(f"Flipping face data {k}")
        #         out.face_data[k] = flip(v)

        # if (p := out.interpolated.points).nelement() > 0:
        #     print("Flipping interpolated points")
        #     out.interpolated.points = flip(p)
        # for k, v in out.interpolated.data.items():
        #     if _is_vector_data(v):
        #         print(f"Flipping interpolated data {k}")
        #         out.interpolated.data[k] = flip(v)

        return out

    def sample_points(
        self,
        n_samples: int,
        set_interpolated: bool = True,
        replacement: bool = True,
        sample_weights: torch.Tensor | str | None = "face areas",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a number of points on each surface. Points are sampled from
        each triangle with a probability proportional to its area.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        set_interpolated: bool
            Set the internal state with the result of the sampling
            (self.interpolated) (default = True).
        replacement : bool, optional
            Whether or not to allow sampling with replacement (default = True).

        Returns
        -------
        samples : torch.Tensor
            Samples with size (n_batch, n_samples, 3).

        References
        ----------
        Smith (2019). Geometrics: Exploiting geometric structure for
            graph-encoded objects.
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/sample_points_from_meshes.py
        """
        if isinstance(sample_weights, torch.Tensor):
            # (n_batch, n_faces)
            assert sample_weights.shape[1] == self.topology.n_faces
        elif sample_weights == "face areas":
            sample_weight = self.compute_face_areas()
        elif sample_weights is None:
            sample_weight = torch.ones(
                self.n_batch, self.topology.n_faces, device=self.get_device()
            )
        else:
            raise ValueError
        sample_weight = sample_weight / sample_weight.sum(1, keepdim=True)

        # Sample faces based on weight (n_batch, n_samples)
        sampled_faces = sample_weight.multinomial(n_samples, replacement)

        # Sample barycentric coordinates for each face
        # (n_batch, n_samples, 3)
        u, w = torch.rand(2, self.n_batch, n_samples, device=self.get_device()).unbind(
            0
        )
        sq_u = u.sqrt()
        sampled_weights = torch.stack(((1 - sq_u), sq_u * (1 - w), sq_u * w), dim=2)

        samples = self.interpolate_vertex_features(
            self.vertices, sampled_faces, sampled_weights
        )
        if set_interpolated:
            self.interpolated.points = samples
            self.interpolated.face_index = sampled_faces
            self.interpolated.weights = sampled_weights
        return samples, sampled_faces, sampled_weights

    def interpolate_vertex_features(self, x, faces, barycentric_coords):
        """Sample a set of features (B, N[, C]) onto the barycentric coordinates
        defined in `barycentric_coords` (B, N_SAMPLES, 3) each of which refers
        to the faces defined in `faces` (B, N_SAMPLES).

        if `x` is not floating point, use nearest neighbor interpolation
        """
        ix = atleast_nd_append(self.batch_ix, 3)
        uv = atleast_nd_append(barycentric_coords, x.ndim + 1)

        vi = self.get_faces()[faces]
        if not x.is_floating_point():
            # select the value(s) associated with the closest vertex
            vi = vi.gather(1, uv.argmin(2, keep_dims=True))
        v = x[ix, vi]
        return torch.sum(v * uv, dim=2) if x.is_floating_point() else v.squeeze(2)

    def vertex_feature_to_face_feature(self, x: torch.Tensor):
        """Compute face features from vertex features by averaging."""
        ix = atleast_nd_append(self.batch_ix, 2)
        return x[ix, self.get_faces()[None]].mean(dim=2)

    def _compute_unnormalized_face_normals(self):
        m = self.as_mesh()
        return torch.cross(m[:, :, 1] - m[:, :, 0], m[:, :, 2] - m[:, :, 0], dim=-1)

    def compute_face_areas(self):
        return 0.5 * self._compute_unnormalized_face_normals().norm(dim=-1)

    def compute_face_normals(self, return_face_areas: bool = False):
        normals = self._compute_unnormalized_face_normals()
        norms = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        normals = normals / norms.clamp_min(min=1e-12)
        return (normals, 0.5 * norms.squeeze(-1)) if return_face_areas else normals

    def compute_vertex_normals(self):
        face_normals = self.compute_face_normals()
        vertex_normals = self._collect_face_values_(face_normals)
        return torch.nn.functional.normalize(vertex_normals, p=2.0, dim=-1)

    def compute_vertex_normals_from_face_normals(self, face_normals: torch.Tensor):
        """Save a computation - perhaps delete."""
        vertex_normals = self._collect_face_values_(face_normals)
        return torch.nn.functional.normalize(vertex_normals, p=2.0, dim=-1)

    def _collect_face_values_(self, values: torch.Tensor):
        buffer = torch.zeros_like(self.vertices)
        buffer = buffer.index_add(1, self.get_faces()[:, 0], values)
        buffer = buffer.index_add(1, self.get_faces()[:, 1], values)
        buffer = buffer.index_add(1, self.get_faces()[:, 2], values)
        return buffer

    def apply_affine(
        self,
        affine: torch.Tensor,
        return_surface: bool = True,
        inplace: bool = False,
        apply_to_vector_data: bool = False,
    ):
        """


        Parameters
        ----------
        affine : torch.Tensor
            _description_
        return_surface : bool, optional
            _description_, by default True
        apply_to_vector_data : bool, optional
            Apply the linear part of the affine transformation (no translation)
            to any vector quantities in vertex, face, or interpolated data.
            If `return_surface = False` and `inplace = False`, this has no
            effect (default = False).
        inplace : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        if inplace:
            self.vertices = apply_affine(affine, self.vertices)
            if apply_to_vector_data:
                self._apply_affine_to_vector_data(affine)
            return self
        else:
            v = apply_affine(affine, self.vertices)
            if return_surface:
                out = Surface(
                    v,
                    self.topology,
                    vertex_data=copy.deepcopy(self.vertex_data),
                    face_data=copy.deepcopy(self.face_data),
                    interpolated_data=copy.deepcopy(self.interpolated),
                )
                if apply_to_vector_data:
                    out._apply_affine_to_vector_data(affine)
                return out
            else:
                return v

    def _apply_affine_to_vector_data(self, affine):
        for k, v in self.vertex_data.items():
            if _is_vector_data(v):
                self.vertex_data[k] = apply_affine(affine, v, translate=False)

        for k, v in self.face_data.items():
            if _is_vector_data(v):
                self.face_data[k] = apply_affine(affine, v, translate=False)

        if _is_vector_data(p := self.interpolated.points):
            self.interpolated.points = apply_affine(affine, p)

        for k, v in self.interpolated.data.items():
            if _is_vector_data(v):
                self.interpolated.data[k] = apply_affine(affine, v, translate=False)

    # def squeeze_batch(self):
    #     self.vertices = self.vertices.squeeze(0)
    #     for k, v in self.vertex_data.items():
    #         self.vertex_data[k] = v.squeeze(0)
    #     for k, v in self.interpolated.items():
    #         if k == "data" and v is not None:
    #             for kk, vv in self.interpolated[k].items():
    #                 self.interpolated[k][kk] = vv.squeeze(0)
    #         elif isinstance(v, torch.Tensor):
    #             self.interpolated[k] = v.squeeze(0)
    #     return self

    # def compute_cotangents(self, eps=1e-8):
    #     """

    #     We use Heron's formula for area and

    #     cot a = (B^2 + C^2 - A^2) / (4 * area)
    #     cot b = (A^2 + C^2 - B^2) / (4 * area)
    #     cot c = (A^2 + B^2 - C^2) / (4 * area)

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     cot :
    #         Cotangents [batch_size, n_triangles, 3] of v0, v1, and v2 in
    #         columns 0, 1, and 2, respectively.

    #     Notes
    #     -----
    #     This function is very similar to pytorch3d.ops.laplacian_matrices.cot_laplacian
    #     """
    #     # mesh  [batch, triangle, 3, coordinates]
    #     # E     [batch, triangle, edge]
    #     # area  [batch, triangle]
    #     # cot   [batch, triangle, cot]

    #     EP = self.topology.edge_pairs
    #     VP = self.topology.vertex_opposite_edge

    #     m = self.as_mesh()

    #     edge_length = torch.stack([m[:, :, i] - m[:, :, j] for i, j in EP], -2).norm(
    #         dim=-1
    #     )
    #     # Heron's formula
    #     s = 0.5 * edge_length.sum(-1)
    #     area = torch.clamp(
    #         s * torch.prod(s[..., None] - edge_length, dim=-1), min=eps
    #     ).sqrt()

    #     # area = self.compute_face_areas()

    #     # cotangents
    #     edge_length_sq = edge_length**2
    #     # cot = torch.stack(
    #     #     [E2[..., i] + E2[..., j] - E2[..., k] for i, j, k in cot_index], -1
    #     # ) / (4 * area[..., None])
    #     cot = torch.stack(
    #         [
    #             edge_length_sq[..., i] + edge_length_sq[..., j] - edge_length_sq[..., k]
    #             for (i, j), k in zip(EP, VP)
    #         ],
    #         -1,
    #     ) / (4 * area[..., None])

    #     return cot, area

    def compute_angles(self, min_edge_length=1e-6, min_angle=1e-3):
        """For each face, compute the angle at each vertex. Optionally,
        integrate the total angle at each vertex

        Theta in Fig. 3 (c) of Meyer (2003).

        Parameters
        ----------
        integrate_vertex_angles
            For each vertex, integrate the corresponding angles for all faces
            which it is part of.


        """
        EI = self.topology.edge_pairs
        VI = self.topology.vertex_edges
        V = self.topology.vertex_opposite_edge

        m = self.as_mesh()
        E = torch.stack([m[:, :, i] - m[:, :, j] for i, j in EI])
        EN = torch.linalg.vector_norm(E, dim=-1).maximum(
            torch.tensor(min_edge_length, device=self.get_device())
        )

        # we need to clamp due to numerical inaccuracies
        face_angles = (
            torch.stack(
                [
                    torch.sum(
                        self.bool_to_sign(EI[ej, 0] == vi)
                        * E[ej]
                        * self.bool_to_sign(EI[ek, 0] == vi)
                        * E[ek],
                        -1,
                    )
                    / (EN[ej] * EN[ek])
                    for vi, (ej, ek) in zip(V, VI)
                ],
                -1,
            )
            .clamp(-1.0, 1.0)
            .acos()
            .clamp(min_angle, torch.pi - min_angle)
        )

        # if face_angles.isnan().any():
        #     raise ValueError(f"NAN is angles: {face_angles.isnan().sum()}")

        # replace zeros
        # min_angle = torch.tensor(1e-3, device=self.device)
        # face_angles = torch.maximum(face_angles, min_angle)

        # replace NaNs
        # Triangles with edges of zero length (!)

        # invalid = torch.any(EN < edge_length_tol, 0)
        # nan_angle = 0.5 * torch.pi - 1e-3 * torch.pi
        # tmp = torch.nan_to_num(face_angles[invalid], nan=nan_angle)
        # face_angles[invalid] = tmp * torch.pi / tmp.sum(-1, keepdim=True)

        return face_angles

    def integrate_face_angles(self, face_angles: torch.Tensor):
        """Integrate face angles to compute the sum of the angles incident on a
        particular vertex.
        """

        return torch.scatter_add(
            torch.zeros(
                (self.n_batch, self.topology.n_vertices), device=self.get_device()
            ),
            1,
            self.get_faces()[:, self.topology.vertex_opposite_edge]
            .long()
            .reshape(1, -1)
            .expand(self.n_batch, -1),
            face_angles.reshape(self.n_batch, -1),
        )

    def compute_cotangents(self, face_angles: torch.Tensor | None = None):
        face_angles = self.compute_angles() if face_angles is None else face_angles
        return 1.0 / face_angles.tan()

    def voronoi_area(
        self, face_angles: torch.Tensor | None = None, apply_correction: bool = True
    ):
        """Calculate Voronoi area (eq. 7) of each vertex or, if
        `apply_correction` is True, calculcate "A_mixed" (fig. 4) from Meyer
        (2003).

        Parameters
        ----------

        Returns
        -------

        References
        ----------
        Meyer (2003). Discrete Differential-Geometry Operator for Triangulated
            2-Manifolds.
        """
        face_angles = self.compute_angles() if face_angles is None else face_angles
        cotangents = self.compute_cotangents(face_angles).reshape(self.n_batch, -1)

        edges = self.topology.get_edges_ravelled()
        edge_len_sq = (
            self.compute_edge_norm().pow(2).reshape(-1, 3 * self.topology.n_faces)
        )

        # The two contributions to the Voronoi area
        # A = (cot_alpha_ij + cot_beta_ij) * (x_i - x_j)
        #   = cot_alpha_ij * (x_i - x_j) + cot_beta_ij * (x_i - x_j)
        cot_x_E2_0 = cotangents * edge_len_sq

        if apply_correction:
            # Each cot_ij * (x_i - x_j) is collected into both x_i and x_j,
            # however, the angle might be obtuse at x_i but not x_j (or vice
            # versa). Hence, we need to duplicate the array: one where x_i is the
            # source vertex and one where x_j is the source vertex
            cot_x_E2_1 = cot_x_E2_0.clone()

            # The Voronoi areas are not valid for obtuse triangles (i.e.,
            # triangles with any angle larger than pi/2). Apply correction
            # (fig. 4)
            is_obtuse_angle = face_angles > torch.pi / 2.0
            is_obtuse_triangle = is_obtuse_angle.any(-1)

            # Get obtuse-ness at each vertex (related to source vertex in last dim)
            obtuse = is_obtuse_angle[..., self.topology.edge_pairs].reshape(
                self.n_batch, -1, 2
            )
            obtuse_tri = (
                is_obtuse_triangle[..., None]
                .expand_as(face_angles)
                .reshape(self.n_batch, -1)
            )

            face_area = self.compute_face_areas()[..., None].expand_as(face_angles)
            face_area = face_area.reshape(self.n_batch, -1)

            # If angle is obtuse at x (vertex of interest), use face_area / 2.0
            # instead of Voronoi area contribution
            # (The factor 4.0 is to compensate for 1.0 / 8.0 later)
            cot_x_E2_0[obtuse[..., 0]] = 0.5 * face_area[obtuse[..., 0]] * 4.0
            cot_x_E2_1[obtuse[..., 1]] = 0.5 * face_area[obtuse[..., 1]] * 4.0

            # If not obtuse at x but triangle is obtuse at any vertex, use
            # face_area / 4.0 instead of Voronoi area contribution
            m = obtuse_tri & ~obtuse[..., 0]
            cot_x_E2_0[m] = 0.25 * face_area[m] * 4.0
            m = obtuse_tri & ~obtuse[..., 1]
            cot_x_E2_1[m] = 0.25 * face_area[m] * 4.0
        else:
            cot_x_E2_1 = cot_x_E2_0

        A_mixed = torch.zeros(
            (self.n_batch, self.topology.n_vertices), device=self.get_device()
        )
        # without correction cot_x_E2_0 == cot_x_E2_1
        A_mixed = torch.index_add(A_mixed, 1, edges[:, 0], cot_x_E2_0)
        A_mixed = torch.index_add(A_mixed, 1, edges[:, 1], cot_x_E2_1)
        A_mixed = A_mixed / 8.0
        return A_mixed

    @staticmethod
    def bool_to_sign(b: bool | torch.Tensor):
        return 1.0 if b else -1.0

    def view_faces_as_vertices(self):
        return self.get_faces()[None].expand((self.n_batch, *self.get_faces().shape))

    def compute_laplace_beltrami_operator(self, f: torch.Tensor | None = None):
        """Computes an estimate of the mean curvature over a function at each
        vertex using discrete Laplace-Beltrami operator. If `f` is None, we
        compute the curvature of the surface itself (i.e., using the vertex
        positions).

        The Laplace-Beltrami operator (also known as the mean curvature
        normal operator)

            K(i) = 2 * H(i) * n(i)
            K(i) = 0.5 * 1.0 / area * sum_{j in N(i)} [ cot(a_ij) + cot(b_ij) ] * (f_i - f_j)

        where N(i) is the neighborhood of i, n(i) is the normal at i, and H(i)
        is the mean curvature. The latter is therefore given by

          H(i) = 0.5 * n(i).T * K(i)  # signed
               = 0.5 * |K(i)|         # unsigned

        Parameters
        ----------
        f : Tensor | None

        Returns
        -------
        _type_
            _description_

        References
        ----------
        Meyer et al. (2003). Discrete Differential-Geometry Operators for
            Triangulated 2-Manifolds.
        https://computergraphics.stackexchange.com/questions/1718/what-is-the-simplest-way-to-compute-principal-curvature-for-a-mesh-triangle

        """
        f = self.vertices if f is None else f
        assert f.shape[0] == self.n_batch
        assert f.shape[1] == self.topology.n_vertices

        angles = self.compute_angles()
        vertex_area = self.voronoi_area(angles)
        cot = self.compute_cotangents(angles)
        cot = cot.reshape(self.n_batch, -1, 1)

        # if cot.isnan().any() or cot.isinf().any() or (cot == 0).any():
        #     print("ANGLE")
        #     print(cot.amin(),cot.amax())
        #     print(angles.amin(),angles.amax())
        #     raise ValueError
        # if (1.0/vertex_area).isnan().any() or vertex_area.isinf().any() or (vertex_area == 0).any():
        #     print("AREA")
        #     print(vertex_area.amin(), vertex_area.amax())
        #     raise ValueError

        edges = self.topology.get_edges_ravelled()
        edge_vec = f[:, edges].diff(dim=-2).squeeze(-2)
        cot_vec_sum = torch.zeros_like(f)
        cot_vec_sum = torch.index_add(cot_vec_sum, 1, edges[:, 0], cot * edge_vec)
        cot_vec_sum = torch.index_add(cot_vec_sum, 1, edges[:, 1], -cot * edge_vec)
        return 0.5 * 1.0 / atleast_nd_append(vertex_area, f.ndim) * cot_vec_sum

    def compute_mean_curvature(self, K: torch.Tensor | None = None, signed=True):
        K = self.compute_laplace_beltrami_operator() if K is None else K
        if signed:
            return 0.5 * torch.sum(self.compute_vertex_normals() * K, -1)
        else:
            return 0.5 * torch.linalg.vector_norm(K, dim=-1)

    def compute_gaussian_curvature(self):
        angles = self.compute_angles()
        vertex_area = self.voronoi_area(angles)
        angles = self.integrate_face_angles(angles)
        return (2 * torch.pi - angles) / vertex_area

    def compute_principal_curvatures(self, H, G, tol=1e-6):
        delta = torch.clamp(H**2 - G, tol).sqrt()
        k1 = H + delta
        k2 = H - delta
        return k1, k2

    def mean_curvature_flow(self, step_size=1.0, n_iter=1, smooth_iter=10):
        v = self.vertices
        for _ in range(n_iter):
            K = self.compute_laplace_beltrami_operator()
            if smooth_iter > 1:
                K = self.compute_iterative_spatial_smoothing(K)
            v = v + step_size * K
        return v

    def compute_iterative_spatial_smoothing(
        self, buffer, iterations=1, dim=1, inplace=False
    ):
        out = buffer if inplace else buffer.clone()

        for _ in range(iterations):
            out.index_reduce_(
                dim,
                self.topology.conv_index_reduce,
                out.index_select(dim, self.topology.conv_index_gather),
                "mean",
                include_self=True,
            )

        return out

    def smooth_taubin(
        self, buffer=None, a=0.8, b=-0.81, n_iter=1, dim=1, inplace=False
    ):
        # assert 0.0 <= a <= 1.0, f"a should be in 0 <= a <= 1 (got {a})"
        # assert b <= -a, f"b should be <= -a (got a = {a} and b = {b})"

        buffer = self.vertices if buffer is None else buffer
        out = buffer if inplace else buffer.clone()

        for _ in range(n_iter):
            # Gauss step
            out = self._smooth_gauss_step(out, a, dim)
            # Taubin step
            out = self._smooth_gauss_step(out, b, dim)

        return out

    def smooth_gauss(self, buffer, a=0.8, n_iter=1, dim=1, inplace=False):
        # assert 0.0 <= a <= 1.0, f"a should be in 0 <= a <= 1 (got {a})"

        out = buffer if inplace else buffer.clone()
        for _ in range(n_iter):
            out = self._smooth_gauss_step(out, a, dim)

        return out

    def _smooth_gauss_step(self, x, a, dim):
        """Perform the following update

            x_i = x_i + a * sum_j (w_ij * (x_j - x_i))  where j in neighborhood of i

        using w_ij = 1/|N_i| where |N_i| is the number of neighbors of i.
        """
        # Compute average over neighbors
        buffer = torch.index_reduce(
            torch.zeros_like(x),
            dim,
            self.topology.conv_index_reduce.long(),  # TODO needs to be long!?
            x.index_select(dim, self.topology.conv_index_gather),
            "mean",
            include_self=False,
        )
        # update
        return x + a * (buffer - x)

    def compute_edge_norm(self, unique: bool = False):
        """ """
        indices = (
            self.topology.get_unique_edges() if unique else self.topology.get_edges()
        )
        edges = self.vertices[:, indices]

        return torch.linalg.vector_norm(edges.diff(dim=-2).squeeze(-2), dim=-1)

    @staticmethod
    def nearest_neighbor_tensors(a: torch.Tensor, b: torch.Tensor):
        # for each element in `a`, this is the index of the closest element
        # in `b`, hence minimum set distance per vertex is
        # dist(a, b[index])

        B, N, _ = a.shape
        size_self = torch.full((B,), N, device=a.device, dtype=torch.int64)
        B, N, _ = b.shape
        size_other = torch.full((B,), N, device=b.device, dtype=torch.int64)

        return cuda_extensions.compute_nearest_neighbor(a, b, size_self, size_other)

    def nearest_neighbor(self, other: "Surface"):
        # for each element in `self`, this is the index of the closest element
        # in `other`, hence minimum set distance per vertex is
        # dist(self.vertices, other.vertices[index])
        return self.nearest_neighbor_tensors(self.vertices, other.vertices)

    def compute_self_intersections(self):
        assert self.vertices.dtype == torch.float
        assert self.get_faces().dtype == torch.int
        vertices = self.vertices
        faces = self.get_faces()

        # the extension returns (intersecting triangles, # intersecting triangles)
        if self.n_batch == 1:
            return cuda_extensions.compute_self_intersections(vertices[0], faces)
        else:
            return [
                cuda_extensions.compute_self_intersections(v, faces) for v in vertices
            ]

    def project_points(
        self,
        points: torch.Tensor,
        tris_per_point: torch.Tensor,
        return_proj: bool = True,
        return_dist: bool = True,
        return_all: bool = False,
    ):
        """Project each point in `points` to the closest point on the surface
        restricted to the triangles in `tris_per_point`.

        PARAMETERS
        ----------
        points : torch.Tensor
            Array with shape (B, N, D) where N is the number of points and D is
            the dimension.
        tris_per_point : torch.Tensor
            If a ragged/nested array, the ith entry contains the triangles against
            which the ith point will be tested.
        return_all : bool
            Whether to return all projection results (i.e., the projection of a
            point on each of the triangles which it was tested against) or only the
            projection on the closest triangle.

        RETURNS
        -------
        tris : ndarray
            The index of the triangle onto which a point was projected.
        weights : ndarray
            The linear interpolation weights resulting in the projection of a point
            onto a particular triangle.
        projs :
            The coordinates of the projection of a point on a triangle.
        dists :
            The distance of a point to its projection on a triangle.

        NOTES
        -----
        The cost function to be minimized is the squared distance between a point
        P and a triangle T

            Q(s,t) = |P - T(s,t)|**2 =
                = a*s**2 + 2*b*s*t + c*t**2 + 2*d*s + 2*e*t + f

        The gradient

            Q'(s,t) = 2(a*s + b*t + d, b*s + c*t + e)

        is set equal to (0,0) to find (s,t).

        REFERENCES
        ----------
        https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

        """
        # if isinstance(pttris, int):
        #     pttris = self.get_closest_triangles(points, pttris, subset)
        # npttris = list(map(len, pttris))
        # pttris = np.concatenate(pttris)

        m = self.as_mesh()
        v0 = m[:, :, 0]  # Origin of the triangle
        e0 = m[:, :, 1] - v0  # s coordinate axis
        e1 = m[:, :, 2] - v0  # t coordinate axis

        # Vector from point to triangle origin (if reverse, the negative
        # determinant must be used)
        w = v0[self.batch_ix, tris_per_point] - points[:, :, None]

        # assert torch.allclose(v0[self.batch_ix, tris_per_point], v0.gather(1, tris_per_point))

        a = torch.sum(e0**2, -1)[self.batch_ix, tris_per_point]
        b = torch.sum(e0 * e1, -1)[self.batch_ix, tris_per_point]
        c = torch.sum(e1**2, -1)[self.batch_ix, tris_per_point]
        d = torch.sum(e0[self.batch_ix, tris_per_point] * w, -1)
        e = torch.sum(e1[self.batch_ix, tris_per_point] * w, -1)
        # f = np.sum(w**2, 1)

        # s,t are so far unnormalized!
        s = b * e - c * d
        t = b * d - a * e
        det = a * c - b**2

        # Project points (s,t) to the closest points on the triangle (s',t')
        sp = torch.zeros_like(s)
        tp = torch.zeros_like(t)

        # We do not need to check a point against all edges/interior of a triangle.
        #
        #          t
        #     \ R2|
        #      \  |
        #       \ |
        #        \|
        #         \
        #         |\
        #         | \
        #     R3  |  \  R1
        #         |R0 \
        #    _____|____\______ s
        #         |     \
        #     R4  | R5   \  R6
        #
        # The code below is equivalent to the following if/else structure
        #
        # if s + t <= 1:
        #     if s < 0:
        #         if t < 0:
        #             region 4
        #         else:
        #             region 3
        #     elif t < 0:
        #         region 5
        #     else:
        #         region 0
        # else:
        #     if s < 0:
        #         region 2
        #     elif t < 0
        #         region 6
        #     else:
        #         region 1

        # Conditions
        st_l1 = (s + t) <= det
        s_l0 = s < 0
        t_l0 = t < 0

        # Region 0 (inside triangle)
        i = torch.where(st_l1 & ~s_l0 & ~t_l0)
        deti = det[i]
        sp[i] = s[i] / deti
        tp[i] = t[i] / deti

        # Region 1
        # The idea is to substitute the constraints on s and t into F(s,t) and
        # solve, e.g., here we are in region 1 and have Q(s,t) = Q(s,1-s) = F(s)
        # since in this case, for a point to be on the triangle, s+t must be 1
        # meaning that t = 1-s.
        i = torch.where(~st_l1 & ~s_l0 & ~t_l0)
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        numer = cc + ee - (bb + dd)
        denom = aa - 2 * bb + cc
        sp[i] = torch.clamp(numer / denom, 0, 1)
        tp[i] = 1 - sp[i]

        # Region 2
        i = torch.where(~st_l1 & s_l0)  # ~t_l0
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        tmp0 = bb + dd
        tmp1 = cc + ee
        j = tmp1 > tmp0
        j_ = ~j
        k = tuple(ii[j] for ii in i)
        k_ = tuple(ii[j_] for ii in i)
        # k, k_ = i[j], i[j_]
        numer = tmp1[j] - tmp0[j]
        denom = aa[j] - 2 * bb[j] + cc[j]
        sp[k] = torch.clamp(numer / denom, 0, 1)
        tp[k] = 1 - sp[k]
        sp[k_] = 0
        tp[k_] = torch.clamp(-ee[j_] / cc[j_], 0, 1)

        # Region 3
        i = torch.where(st_l1 & s_l0 & ~t_l0)
        cc, ee = c[i], e[i]
        sp[i] = 0
        tp[i] = torch.clamp(-ee / cc, 0, 1)

        # Region 4
        i = torch.where(st_l1 & s_l0 & t_l0)
        aa, cc, dd, ee = a[i], c[i], d[i], e[i]
        j = dd < 0
        j_ = ~j
        k = tuple(ii[j] for ii in i)
        k_ = tuple(ii[j_] for ii in i)
        # k, k_ = i[j], i[j_]
        sp[k] = torch.clamp(-dd[j] / aa[j], 0, 1)
        tp[k] = 0
        sp[k_] = 0
        tp[k_] = torch.clamp(-ee[j_] / cc[j_], 0, 1)

        # Region 5
        i = torch.where(st_l1 & ~s_l0 & t_l0)
        aa, dd = a[i], d[i]
        tp[i] = 0
        sp[i] = torch.clamp(-dd / aa, 0, 1)

        # Region 6
        i = torch.where(~st_l1 & t_l0)  # ~s_l0
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        tmp0 = bb + ee
        tmp1 = aa + dd
        j = tmp1 > tmp0
        j_ = ~j
        k = tuple(ii[j] for ii in i)
        k_ = tuple(ii[j_] for ii in i)
        # k, k_ = i[j], i[j_]
        numer = tmp1[j] - tmp0[j]
        denom = aa[j] - 2 * bb[j] + cc[j]
        tp[k] = torch.clamp(numer / denom, 0, 1)
        sp[k] = 1 - tp[k]
        tp[k_] = 0
        sp[k_] = torch.clamp(-dd[j_] / aa[j_], 0, 1)

        projs = (
            v0[self.batch_ix, tris_per_point]
            + sp[..., None] * e0[self.batch_ix, tris_per_point]
            + tp[..., None] * e1[self.batch_ix, tris_per_point]
        )
        # Distance from original point to its projection on the triangle
        dists = torch.linalg.norm(points[:, :, None] - projs, dim=-1)

        if return_all:
            tris = tris_per_point
            weights = torch.stack((1 - sp - tp, sp, tp), dim=-1)
        else:
            # Find the closest projection
            i = dists.argmin(-1)

            ind = torch.arange(points.shape[1], device=points.device)

            tris = tris_per_point[self.batch_ix, ind, i]
            spi = sp[self.batch_ix, ind, i]
            tpi = tp[self.batch_ix, ind, i]
            weights = torch.stack((1 - spi - tpi, spi, tpi), dim=-1)
            if return_dist:
                dists = dists[self.batch_ix, ind, i]
            if return_proj:
                projs = projs[self.batch_ix, ind, i]

        if return_dist:
            if return_proj:
                return tris, weights, projs, dists
            else:
                return tris, weights, dists
        elif return_proj:
            return tris, weights, projs
        else:
            return tris, weights

    def normalize_to_bounding_box(self):
        """Normalize coordinates by centering the surface on the origin and
        dividing by the maximum along in each dimension.
        """
        self.center_on_origin()
        size = self.bounding_box()[:, [1]]  # .amax()
        self.vertices = self.vertices / size

    def rotate(self, k: torch.Tensor, alpha: torch.Tensor, inplace: bool = False):
        """Rotate `v` by `alpha` (angle) around `k` (axis).

        Rodrigues' rotation formula.

        Parameters
        ----------
        k : torch.Tensor
            Axis around which to rotate. Either one axis for all vertices
            (k.shape = (3,)) or one axis per vertex (k.shape = (..., 3)).
        alpha : torch.Tensor


        References
        ----------
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

        """
        res = rotate(self.vertices, k, alpha)
        if inplace:
            self.vertices = res
        return res

    def rotate_dim(self, dim, alpha, inplace: bool = False):
        """Rotate around one of the major axes as specified by `dim`."""
        cos_angle = alpha.cos()

        n = self.n_dim
        k = torch.zeros(n, device=self.get_device())
        k[dim] = 1.0
        k_as_v = k.expand_as(self.vertices)

        q = torch.zeros_like(self.vertices)
        q[..., dim] = self.vertices[..., dim]

        res = (
            self.vertices * cos_angle
            + torch.linalg.cross(self.vertices, k_as_v) * alpha.sin()
            + q * (1 - cos_angle)
        )
        if inplace:
            self.vertices = res
        else:
            return res


class Lines(Surface):
    def __init__(
        self,
        vertices: torch.Tensor,
        topology: brainnet.mesh.topology.Topology | torch.Tensor,
        topology_class: str = "LineTopology",
        vertex_data=None,
        face_data=None,
        interpolated_data=None,
    ):
        super().__init__(
            vertices,
            topology,
            topology_class,
            vertex_data,
            face_data,
            interpolated_data,
        )

    def compute_line_lengths(self):
        edges = self.as_mesh().diff(dim=2).squeeze(2)
        return torch.linalg.vector_norm(edges, dim=-1)

    def sample_points(
        self,
        n_samples: int,
        set_interpolated: bool = True,
        replacement: bool = True,
        sample_weights: torch.Tensor | str | None = "line lengths",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(sample_weights, torch.Tensor):
            # (n_batch, n_faces)
            assert sample_weights.shape[1] == self.topology.n_faces
        elif sample_weights == "line lengths":
            sample_weight = self.compute_line_lengths()
        elif sample_weights is None:
            sample_weight = torch.ones(
                self.n_batch, self.topology.n_faces, device=self.get_device()
            )
        else:
            raise ValueError
        sample_weight = sample_weight / sample_weight.sum(1, keepdim=True)

        # Sample faces based on weight (n_batch, n_samples)
        sampled_lines = sample_weight.multinomial(n_samples, replacement)

        # Sample barycentric coordinates for each face
        # (n_batch, n_samples, 3)
        u = torch.rand(self.n_batch, n_samples, device=self.get_device())
        sampled_weights = torch.stack((u, 1.0 - u), dim=2)

        samples = self.interpolate_vertex_features(
            self.vertices, sampled_lines, sampled_weights
        )
        if set_interpolated:
            self.interpolated.points = samples
            self.interpolated.face_index = sampled_lines
            self.interpolated.weights = sampled_weights
        return samples, sampled_lines, sampled_weights


def load_deepsurfer_template(
    subdivision: int, surface: str, hemi: list[str] | tuple = ("lh", "rh")
):
    """Load DeepSurfer template surface at `subdivision` level."""
    assert (
        0 <= subdivision <= 6
    ), "DeepSurfer template is only defined at resolution levels 0 to 6."
    assert len(hemi) in {1, 2}

    topo_lh = brainnet.mesh.topology.DeepSurferTopology.recursive_subdivision(
        subdivision
    )[-1]
    topo = {}
    if "lh" in hemi:
        topo["lh"] = topo_lh
    if "rh" in hemi:
        topo["rh"] = copy.deepcopy(topo_lh)
        topo["rh"].reverse_face_orientation()

    template = brainsynth.resources.Template()
    return torch.nn.ModuleDict(
        {
            h: Surface(
                template.load_surface(h, surface)["vertices"][: v.n_vertices],
                v,
            )
            for h, v in topo.items()
        }
    )
