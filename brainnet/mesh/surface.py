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

    def as_mesh(self):
        # (n_batch, n_vertices, v_per_face, coordinates)
        return self.vertices[:, self.faces]

    def compute_face_barycenters(self):
        return self.as_mesh().mean(2)

    def bounding_box(self):
        """(batch, 2, 3)."""
        return torch.stack((self.vertices.amin(1), self.vertices.amax(1)), dim=1)

    def center_on_origin(self):
        center = self.bounding_box().mean(1)[:, None]
        self.vertices = self.vertices - center

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

    def interpolate_vertex_features(self, x, faces, barycentric_coords):
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

    def mean_curvature_flow(self, step_size=1.0, n_iter=1, smooth_iter=10):
        v = self.vertices
        for _ in range(n_iter):
            K = self.compute_laplace_beltrami_operator()
            if smooth_iter > 1:
                K = self.compute_iterative_spatial_smoothing(K)
            v = v + step_size * K
        return v

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

    def smooth_taubin(
        self, buffer=None, a=0.8, b=-0.85, n_iter=1, dim=1, inplace=False
    ):
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
        """ """
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
        assert self.faces.dtype == torch.int  # torch.int64
        vertices = self.vertices.detach()
        faces = self.faces.detach()

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
        m = self.as_mesh()
        v0 = m[:, :, 0]  # Origin of the triangle
        e0 = m[:, :, 1] - v0  # s coordinate axis
        e1 = m[:, :, 2] - v0  # t coordinate axis

        # Vector from point to triangle origin (if reverse, the negative
        # determinant must be used)
        w = v0[self.batch_ix, tris_per_point] - points[:, :, None]

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
        cos_angle = alpha.cos()

        k = torch.nn.functional.normalize(k, dim=-1)
        k_as_v = k.expand_as(self.vertices)

        res = (
            self.vertices * cos_angle
            + torch.cross(self.vertices, k_as_v) * alpha.sin()
            + torch.sum(self.vertices * k_as_v, dim=-1, keepdim=True) * k_as_v * (1 - cos_angle)
        )
        if inplace:
            self.vertices = res
        else:
            return res

    def rotate_dim(self, dim, alpha, inplace: bool = False):
        """Rotate around one of the major axes as specified by `dim`."""
        cos_angle = alpha.cos()

        n = self.n_dim
        k = torch.zeros(n, device=self.device)
        k[dim] = 1.0
        k_as_v = k.expand_as(self.vertices)

        q = torch.zeros_like(self.vertices)
        q[..., dim] = self.vertices[..., dim]

        res = (
            self.vertices * cos_angle
            + torch.cross(self.vertices, k_as_v) * alpha.sin()
            + q * (1 - cos_angle)
        )
        if inplace:
            self.vertices = res
        else:
            return res
