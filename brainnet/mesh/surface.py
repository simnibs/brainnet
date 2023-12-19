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
    if t.ndim == n:
        return t
    else:
        return atleast_nd_prepend(t[None], n)

def atleast_nd_append(t, n):
    if t.ndim == n:
        return t
    else:
        return atleast_nd_prepend(t[..., None], n)

class BatchedSurfaces:
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
        assert value.shape[1] == self.topology.n_vertices, f"Vertices dimension mismatch: {value.shape[1]} and {self.topology.n_vertices}"
        self._vertices = value
        self.n_batch, _, self.n_dim = self._vertices.shape

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

    def compute_face_normals(self, return_face_areas: bool = False):
        mesh = self.vertices[:, self.faces]
        normals = torch.cross(
            mesh[:, :, 1] - mesh[:, :, 0], mesh[:, :, 2] - mesh[:, :, 0], dim=-1
        )
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
        self, face_normals: torch.Tensor, n_vertices,
    ):
        """Save a computation - perhaps delete."""
        vertex_normals = torch.zeros((face_normals.shape[0], n_vertices, 3), device=self.vertices.device)
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
        ) / 4 * area[..., None]

        return cot, area

    def view_faces_as_vertices(self):
        return self.faces[None].expand((self.n_batch, *self.faces.shape))


    def compute_mean_curvature_vector(self):
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
        vertex_area = torch.zeros((self.n_batch, self.topology.n_vertices), device=self.vertices.device)
        vertex_area.index_add_(
            1, self.faces.ravel(), face_area.reshape(self.n_batch, -1)
        )
        inv_vertex_area = 1 / vertex_area

        # The edges corresponding to the values of `cotangent`
        # edge0 = faces[:, (1, 0, 0)].ravel()
        # edge1 = faces[:, (2, 2, 1)].ravel()
        # edge0 = edge0[None].expand(n_batch, -1)
        # edge1 = edge0[None].expand(n_batch, -1)

        n_edges = 3*self.topology.n_faces
        edges = self.topology.edges_from_faces()

        # for each vertex, sum the cotangents of all of its edges weighted by
        # the vertex itself, i.e.,
        #
        #   sum_{j in N(i)} (cot(a_ij) + cot(b_ij)) * f(vi)
        #   = f(vi) * sum_{j in N(i)} cot(a_ij) + cot(b_ij)
        #
        # where N(i) is the 1-ring neighbors of vertex i.
        cot_ab_sum = torch.zeros(
            (self.n_batch, self.topology.n_vertices),
            device=self.vertices.device
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
        #   K(v) = 2* H(v) * n(v)
        #   K(v) = 0.5 * 1/area * sum_{j in N(i)} [cot(a_ij))+cot(b_ij)] * (v_j - v_i)
        #
        # where H(v) is the mean curvature and n(v) is the normals at v, thus
        #
        #   H(v) = 0.5 * n(v).T * K(v)  # signed
        #        = 0.5 * |K(v)|         # unsigned
        #
        return 0.5 * inv_vertex_area[..., None] * (cot_ab_vj - cot_ab_vi)


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


    def compute_iterative_spatial_smoothing(self, buffer, iterations=1, dim=1, inplace=False):
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
                include_self=True
            )
        return None if inplace else out


    def compute_edge_lengths(self):
        """Variance of normalized edge length. Edge lengths are normalized so that
        their sum is equal to the number of edges. Otherwise, zero error can be
        achieved simply by shrinking the mesh.

        The idea is to encourage equilateral triangles.

        Parameters
        ----------
        surfaces : BatchedSurfaces


        Returns
        -------
        loss : float

        """
        return self.vertices[:, self.topology.vertex_adjacency].diff(dim=-2).squeeze(-2).norm(dim=-1)


    # def matched_distance(self, other: "BatchedSurfaces", index=None):
    #     other_vertices = other.vertices
    #     if index is not None:
    #         other_vertices = other_vertices[index]
    #     return torch.norm(self.vertices - other_vertices, dim=-1)


    def nearest_neighbor(self, other: "BatchedSurfaces"):
        # for each element in `self`, this is the index of the closest element
        # in `other`, hence minimum set distance per vertex is
        # dist(self.vertices, other.vertices[index])

        B, N, _ = self.vertices.shape
        size_self = torch.full((B, ), N, device=self.vertices.device, dtype=torch.int64)
        B, N, _ = other.vertices.shape
        size_other = torch.full((B, ), N, device=other.vertices.device, dtype=torch.int64)

        return cuda_extensions.compute_nearest_neighbor(
            self.vertices, other.vertices, size_self, size_other
        )


    # def chamfer_distance(self, other: "BatchedSurfaces"):
    #     """

    #     Parameters
    #     ----------


    #     Returns
    #     -------
    #     dist :
    #         (Squared) distance between closest points in point sets `a` and `b`.
    #     index_other :
    #         This array gives, for each element in `self.vertices`, the index of
    #         the closest element in `other.vertices`.
    #     index_self :
    #         This array gives, for each element in `other.vertices`, the index
    #         of the closest element in `self.vertices`.
    #     """
    #     # for each element in `self`, this is the index of the closest element
    #     # in `other`
    #     index_other = cuda_extensions.compute_nearest_neighbor(self.vertices, other.vertices)
    #     dist = torch.norm(self.vertices - other.vertices[index_other], dim=-1)

    #     return dist, index_other



    def compute_self_intersections(self):
        assert self.vertices.dtype == torch.float
        assert self.faces.dtype == torch.int64
        vertices = self.vertices.detach()
        faces = self.faces.detach()

        # the extension returns (intersecting triangles, # intersecting triangles)
        return torch.stack(
            [cuda_extensions.compute_self_intersections(v, faces)[0] for v in vertices]
        )#.nonzero().squeeze()
