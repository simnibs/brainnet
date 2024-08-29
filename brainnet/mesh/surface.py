import torch

from brainnet.mesh.topology import Topology

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


def atleast_nd_prepend(t, n):
    if t.ndim >= n:
        return t
    else:
        return atleast_nd_prepend(t[None], n)


def atleast_nd_append(t, n):
    if t.ndim >= n:
        return t
    else:
        return atleast_nd_append(t[..., None], n)


class TemplateSurfaces:
    def __init__(
        self,
        vertices: torch.Tensor,
        topology: Topology | torch.Tensor,
    ) -> None:
        """A batch of surfaces (vertices) that share a common topology.

        Parameters
        ----------
        vertices : torch.Tensor
            _description_
        topology : Topology | torch.Tensor
            If a tensor, then it is assumed to be a 2d array representing the
            connectivity of the surface.
        """
        self.device = vertices.device

        self.topology = topology
        self.vertices = vertices
        self.vertex_data = {}
        self.face_data = {}

    @property
    def topology(self):
        return self._topology

    @topology.setter
    def topology(self, value):
        self._topology = value if isinstance(value, Topology) else Topology(value)
        self.faces = self._topology.faces

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        value = atleast_nd_prepend(value, 3)
        assert (
            value.shape[1] == self.topology.n_vertices
        ), f"Vertices dimension mismatch: {value.shape[1]} and {self.topology.n_vertices}"
        self._vertices = value
        self.n_batch, _, self.n_dim = self._vertices.shape
        # batch indexer (e.g., [[0, 1, 2]])
        self.batch_ix = torch.arange(self.n_batch, device=self.device)

        # if hasattr(self, "mean_curvature_vector"):
        #     self.mean_curvature_vector = self.compute_mean_curvature_vector()

    # @property
    # def mean_curvature_vector(self):
    #     return self._mean_curvature_vector

    # @mean_curvature_vector.setter
    # def mean_curvature_vector(self, value):
    #     value = torch.atleast_3d(value)
    #     assert value.shape[1] == self.topology.n_vertices
    #     self._mean_curvature_vector = value

    def sample_points(
        self,
        n_samples: int,
        replacement=True,
        weights: torch.Tensor | None = None,
        return_sampled_faces_and_bc=False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a number of points on each surface. Points are sampled from
        each triangle with a probability proportional to its area.

        Parameters
        ----------
        n_samples : int
            _description_
        replacement : bool, optional
            _description_, by default True

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
        sample_weight = self.compute_face_areas()
        sample_weight = (
            sample_weight * weights if weights is not None else sample_weight
        )
        sample_weight = sample_weight / sample_weight.sum(1)[:, None]

        # Sample faces based on weight
        # (n_batch, n_samples)
        sampled_faces = sample_weight.multinomial(n_samples, replacement)

        # Sample barycentric coordinates for each face
        # (n_batch, n_samples, 3)
        u, w = torch.rand(2, self.n_batch, n_samples, device=self.device).unbind(0)
        sq_u = u.sqrt()
        sampled_coords = torch.stack(((1 - sq_u), sq_u * (1 - w), sq_u * w), dim=2)

        samples = self.interpolate_vertex_features(
            self.vertices, sampled_faces, sampled_coords
        )
        if return_sampled_faces_and_bc:
            return samples, sampled_faces, sampled_coords
        else:
            return samples

    def interpolate_vertex_features(
        self, x, faces, barycentric_coords
    ):
        """Sample a set of features (B, N[, C]) onto the barycentric coordinates
        defined in `barycentric_coords` (B, N_SAMPLES, 3) each of which refers
        to the faces defined in `faces` (B, N_SAMPLES).
        """
        ix = atleast_nd_append(self.batch_ix, 3)
        uv = atleast_nd_append(barycentric_coords, x.ndim + 1)
        return torch.sum(x[ix, self.faces[faces]] * uv, dim=2)

    def vertex_feature_to_face_feature(self, x: torch.Tensor):
        """Compute face features from vertex features by averaging."""
        ix = atleast_nd_append(self.batch_ix, 2)
        return x[ix, self.faces[None]].mean(dim=2)

    def _compute_unnormalized_face_normals(self):
        mesh = self.vertices[:, self.faces]
        return torch.cross(
            mesh[:, :, 1] - mesh[:, :, 0], mesh[:, :, 2] - mesh[:, :, 0], dim=-1
        )

    def compute_face_areas(self):
        return 0.5 * self._compute_unnormalized_face_normals().norm(dim=-1)

    def compute_face_normals(self, return_face_areas: bool = False):
        normals = self._compute_unnormalized_face_normals()
        if return_face_areas:
            norms = normals.norm(dim=-1)
            face_areas = 0.5 * norms
            normals = normals / norms.clamp_min(min=1e-12)[..., None]
            return normals, face_areas
        else:
            return torch.nn.functional.normalize(normals, p=2.0, dim=-1)

    def compute_vertex_normals(self):
        face_normals = self.compute_face_normals()

        vertex_normals = torch.zeros_like(self.vertices)
        self._collect_face_values_(vertex_normals, face_normals)

        return torch.nn.functional.normalize(vertex_normals, p=2.0, dim=-1)

    def compute_vertex_normals_from_face_normals(
        self,
        face_normals: torch.Tensor,
        n_vertices,
    ):
        """Save a computation - perhaps delete."""
        vertex_normals = torch.zeros(
            (face_normals.shape[0], n_vertices, 3), device=self.device
        )
        self._collect_face_values_(vertex_normals, face_normals)

        return torch.nn.functional.normalize(vertex_normals, p=2.0, dim=-1)

    def _collect_face_values_(self, buffer: torch.Tensor, values: torch.Tensor):
        buffer.index_add_(1, self.faces[:, 0], values)
        buffer.index_add_(1, self.faces[:, 1], values)
        buffer.index_add_(1, self.faces[:, 2], values)

    def compute_cotangents(self, eps=1e-8):
        """

        We use Heron's formula for area and

        cot a = (B^2 + C^2 - A^2) / (4 * area)
        cot b = (A^2 + C^2 - B^2) / (4 * area)
        cot c = (A^2 + B^2 - C^2) / (4 * area)


        Parameters
        ----------


        Returns
        -------
        cot :
            Cotangents [batch_size, n_triangles, 3] of v0, v1, and v2 in
            columns 0, 1, and 2, respectively.

        Notes
        -----
        This function is very similar to pytorch3d.ops.laplacian_matrices.cot_laplacian
        """
        edge_index = ((1, 2), (0, 2), (0, 1))  # opposite v0, v1, v2, respectively
        cot_index = ((1, 2, 0), (0, 2, 1), (0, 1, 2))

        # mesh  [batch, triangle, 3, coordinates]
        # E     [batch, triangle, edge]
        # area  [batch, triangle]
        # cot   [batch, triangle, cot]

        mesh = self.vertices[:, self.faces]

        # edge lengths
        E = torch.stack(
            [mesh[..., i, :] - mesh[..., j, :] for i, j in edge_index], -2
        ).norm(dim=-1)
        # Heron's formula
        s = 0.5 * E.sum(-1)
        area = torch.clamp(s * torch.prod(s[..., None] - E, dim=-1), min=eps).sqrt()
        # cotangents
        E2 = E**2
        cot = torch.stack(
            [E2[..., i] + E2[..., j] - E2[..., k] for i, j, k in cot_index], -1
        ) / (4 * area[..., None])

        return cot, area

    def view_faces_as_vertices(self):
        return self.faces[None].expand((self.n_batch, *self.faces.shape))

    def compute_laplace_beltrami_operator(self):
        """Computes a discrete estimate of the mean curvature at each vertex.

        Using discrete Laplace-Beltrami operator

        Here we compute the curvature of the surface itself but we could also
        do so for a function defined on the surface.


        Parameters
        ----------
        y_pred : _type_
            _description_
        order : _type_
            _description_

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
        cotangent, face_area = self.compute_cotangents()

        # Compute area per vertex
        face_area = face_area / 3.0
        face_area = face_area[..., None].expand((self.n_batch, *self.faces.shape))
        vertex_area = torch.zeros(
            (self.n_batch, self.topology.n_vertices), device=self.device
        )
        vertex_area.index_add_(
            1, self.faces.ravel(), face_area.reshape(self.n_batch, -1)
        )
        inv_vertex_area = 1 / vertex_area

        # The edges corresponding to the values of `cotangent`
        # edge0 = faces[:, (1, 0, 0)].ravel()
        # edge1 = faces[:, (2, 2, 1)].ravel()
        # edge0 = edge0[None].expand(n_batch, -1)
        # edge1 = edge0[None].expand(n_batch, -1)

        n_edges = 3 * self.topology.n_faces
        edges = self.topology.edges_from_faces()

        # for each vertex, sum the cotangents of all of its edges weighted by
        # the vertex itself, i.e.,
        #
        #   sum_{j in N(i)} (cot(a_ij) + cot(b_ij)) * f(vi)
        #   = f(vi) * sum_{j in N(i)} cot(a_ij) + cot(b_ij)
        #
        # where N(i) is the 1-ring neighbors of vertex i.
        cot_ab_sum = torch.zeros(
            (self.n_batch, self.topology.n_vertices), device=self.device
        )
        cot_ab_sum.index_add_(
            1,
            edges[:, 0],
            cotangent.reshape(self.n_batch, n_edges),
        )
        cot_ab_sum.index_add_(
            1,
            edges[:, 1],
            cotangent.reshape(self.n_batch, n_edges),
        )

        cot_ab_vi = cot_ab_sum[..., None] * self.vertices

        # for each vertex, again compute the sum of cotangents of each edge,
        # but this time weigh by the opposite vertex and sum over all edges,
        # i.e.
        #
        #   sum_{j in N(i)} (cot(a_ij) + cot(b_ij) * f(vj)

        cotangent = cotangent.reshape(self.n_batch, -1, 1)

        # NOTE mind the opposite edge indexing!
        cot_ab_vj = torch.zeros_like(self.vertices)
        cot_ab_vj.index_add_(
            1, edges[..., 0], cotangent * self.vertices[:, edges[..., 1]]
        )
        cot_ab_vj.index_add_(
            1, edges[..., 1], cotangent * self.vertices[:, edges[..., 0]]
        )

        # Finally, the Laplace-Beltrami operator (also known as the mean
        # curvature normal operator)
        #
        #   K(v) = 2 * H(v) * n(v)
        #   K(v) = 0.5 * 1/area * sum_{j in N(i)} [cot(a_ij))+cot(b_ij)] * (v_j - v_i)
        #
        # where H(v) is the mean curvature and n(v) are the normals at v, thus
        #
        #   H(v) = 0.5 * n(v).T * K(v)  # signed
        #        = 0.5 * |K(v)|         # unsigned
        #
        return 0.5 * inv_vertex_area[..., None] * (cot_ab_vj - cot_ab_vi)

    def compute_mean_curvature(self, K, signed=True):
        if signed:
            return 0.5 * torch.sum(self.compute_vertex_normals() * K, -1)
        else:
            return 0.5 * K.norm(dim=-1)

    # def compute_curvatures(self, K):

    #     # Mean curvature
    #     H = 0.5 * K.norm(dim=-1)

    #     # Gaussian curvature
    #     # angles of each face as incident on a particular vertex
    #     vertex_angles = ...
    #     G = (2 * torch.pi - vertex_angles) / face_area

    #     # Principal curvatures
    #     eps = 1e-8
    #     sq_factor = np.sqrt(torch.clamp(H**2 - G, min=eps))
    #     k1 = H + sq_factor
    #     k2 = H - sq_factor

    #     # elif method == "sphere-approx":
    #     # # Edge curvature
    #     # nn = compute_normals(y_pred, self.topology[order]["faces"])
    #     # edge_vec = y_pred[reduce_idx] - y_pred[edge_idx]
    #     # norm_vec = nn[reduce_idx] - nn[edge_idx]
    #     # curv_edge = norm_vec @ edge_vec / edge_vec.pow(2).sum(-1)

    #     # # Vertex curvature
    #     # curv = torch.zeros(len(y_pred))
    #     # curv = curv.index_reduce(0, reduce_idx, curv_edge, reduce="mean", include_self=False)
    #     # # curv.scatter_reduce(-1, reduce_idx, curv_edge, reduce="mean")

    def compute_iterative_spatial_smoothing(
        self, buffer, iterations=1, dim=1, inplace=False
    ):
        reduce_index, gather_index = self.topology.get_convolution_indices()
        if inplace:
            out = buffer
        else:
            out = torch.zeros_like(buffer, device=self.vertices.device)
            out.copy_(buffer)
        for _ in range(iterations):
            out.index_reduce_(
                dim,
                reduce_index,
                out.index_select(dim, gather_index),
                "mean",
                include_self=True,
            )
        return out

    def smooth_taubin(self, buffer=None, a=0.8, b=-0.85, n_iter=1, dim=1, inplace=False):
        # assert 0.0 <= a <= 1.0, f"a should be in 0 <= a <= 1 (got {a})"
        # assert b <= -a, f"b should be <= -a (got a = {a} and b = {b})"

        reduce_index, gather_index = self.topology.get_convolution_indices()

        buffer = self.vertices if buffer is None else buffer

        if inplace:
            out = buffer
        else:
            out = torch.zeros_like(buffer, device=self.vertices.device)
            out.copy_(buffer)

        for _ in range(n_iter):
            # Gauss step
            out = self._smooth_gauss_step(out, a, dim, reduce_index, gather_index)
            # Taubin step
            out = self._smooth_gauss_step(out, b, dim, reduce_index, gather_index)

        return out

    # for i in (0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99,0.999,1):
    #     print(f"{i:3f} : {H[0].quantile(i)}")

    def smooth_gauss(self, buffer, a=0.8, n_iter=1, dim=1, inplace=False):
        # assert 0.0 <= a <= 1.0, f"a should be in 0 <= a <= 1 (got {a})"

        reduce_index, gather_index = self.topology.get_convolution_indices()

        if inplace:
            out = buffer
        else:
            out = torch.zeros_like(buffer, device=buffer.device)
            out.copy_(buffer)

        for _ in range(n_iter):
            out = self._smooth_gauss_step(out, a, dim, reduce_index, gather_index)

        return out

    @staticmethod
    def _smooth_gauss_step(x, a, dim, reduce_index, gather_index):
        """Perform the following update

            x_i = x_i + a * sum_j (w_ij * (x_j - x_i))  where j in neighborhood of i

        using w_ij = 1/|N_i| where |N_i| is the number of neighbors of i.
        """
        # Compute average over neighbors
        buffer = torch.zeros_like(x, device=x.device)
        buffer.index_reduce_(
            dim,
            reduce_index,
            x.index_select(dim, gather_index),
            "mean",
            include_self=False,
        )
        # update
        return x + a * (buffer - x)


    def compute_edge_lengths(self):
        """Variance of normalized edge length. Edge lengths are normalized so that
        their sum is equal to the number of edges. Otherwise, zero error can be
        achieved simply by shrinking the mesh.

        The idea is to encourage equilateral triangles.

        Parameters
        ----------
        surfaces : TemplateSurfaces


        Returns
        -------
        loss : float

        """
        return (
            self.vertices[:, self.topology.vertex_adjacency]
            .diff(dim=-2)
            .squeeze(-2)
            .norm(dim=-1)
        )

    # def matched_distance(self, other: "BatchedSurfaces", index=None):
    #     other_vertices = other.vertices
    #     if index is not None:
    #         other_vertices = other_vertices[index]
    #     return torch.norm(self.vertices - other_vertices, dim=-1)

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

    def nearest_neighbor(self, other: "TemplateSurfaces"):
        # for each element in `self`, this is the index of the closest element
        # in `other`, hence minimum set distance per vertex is
        # dist(self.vertices, other.vertices[index])
        return self.nearest_neighbor_tensors(self.vertices, other.vertices)

    def compute_self_intersections(self):
        assert self.vertices.dtype == torch.float
        assert self.faces.dtype == torch.int # torch.int64
        vertices = self.vertices.detach()
        faces = self.faces.detach()

        # the extension returns (intersecting triangles, # intersecting triangles)
        if vertices.shape[0] == 1:
            return cuda_extensions.compute_self_intersections(vertices[0], faces)
        else:
            return [
                cuda_extensions.compute_self_intersections(v, faces)[0]
                for v in vertices
            ]
