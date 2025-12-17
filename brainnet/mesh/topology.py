import torch


class Topology(torch.nn.Module):
    def __init__(
        self,
        faces: torch.Tensor,
        edge_pairs: torch.Tensor | None = None,
        retain_edge_order: bool = True,
    ):
        super().__init__()

        self.register_buffer("faces", faces, persistent=False)
        self.subdivision_factor = 4
        self.dtype = self.faces.dtype
        self.n_faces = self.faces.shape[0]
        self.vertices_per_face = self.faces.shape[1]
        self.n_vertices = self.n_faces_to_n_vertices()
        # self._reversed_face_order = (0, 2, 1)

        if edge_pairs is None:
            edge_pairs = torch.tensor(
                [[0, 1], [1, 2], [2, 0]], dtype=torch.int, device=self.device()
            )
        else:
            edge_pairs = edge_pairs.to(self.device())
        self.register_buffer("edge_pairs", edge_pairs, persistent=False)

        self.set_topology_information(retain_edge_order)

    def device(self):
        return self.faces.device

    def reverse_face_orientation(self):
        # self.faces = self.faces[:, self._reversed_face_order]
        self.faces = self.faces.flip(-1)
        # Ensure subdivide_faces is still valid
        self.faces_to_edges = self.faces_to_edges.flip(-1)

    def get_edges(self, sort: bool = False):
        # (n_faces, 3, 2)
        edges = torch.stack(
            [
                self.faces[:, self.edge_pairs[:, 0]],
                self.faces[:, self.edge_pairs[:, 1]],
            ],
            dim=2,
        )
        return edges.sort(-1)[0] if sort else edges

    def get_unique_edges(self):
        return self.vertex_adjacency

    def get_edges_ravelled(
        self,
        sort_between: bool = False,
        sort_within: bool = False,
    ):
        """

        sort :
            Sort between edges.
        sort_within :
            Sort vertex indices within each edge.
        """
        edges = self.get_edges(sort_within).reshape((-1, 2))
        if sort_between:
            edge0, edge0_index = edges[:, 0].sort()
            edges[:, 0] = edge0
            edges[:, 1] = edges[edge0_index, 1]
        return edges

    def unpool(self, features, reduce="amax"):
        assert (
            self.n_vertices == features.shape[-1]
        ), f"Expected {self.n_vertices} to unpool from (got {features.shape[-1]})."
        # features sampled in between all edges (i.e., where the new vertices)
        # are added
        up_features = features[..., self.vertex_adjacency]
        up_features = getattr(up_features, reduce)(-1)

        return torch.cat([features, up_features], dim=-1)

    def pool(self, features: torch.Tensor, reduce="amax"):
        """Pool values from neighbors into each vertex."""
        assert (
            self.n_vertices == features.shape[-1]
        ), f"Expected {self.n_vertices} to pool from (got {features.shape[-1]})."

        # pool from `gather_index` into `reduce_index`

        # Pool
        # (1) from the vertices which also exist on the lower level into
        #     themselves
        # (2) from the center vertex into each vertex making up that edge in
        #     the previous level

        # we take advantage of the fact that the edges are sorted along both
        # dimensions, hence the first n_edges//2 edges are the ones from the
        # target vertices (subsampled) to the source vertices
        # (from the upsampling procedure)
        return torch.index_reduce(
            self.subsample_array(features),
            -1,
            self.pool_index_reduce,
            features[..., self.pool_index_gather],
            reduce,
            include_self=True,
        )

    def subsample_array(self, arr: torch.Tensor, dim: int = -1):
        """ """
        assert self.n_vertices == arr.shape[2]
        n = self.n_vertices_lower_order()

        return arr.narrow(dim, 0, n)  # or narrow_copy?

    def get_vertexwise_faces(self):
        """

        The faces associated with vertex i is given by

            faces_for_vertex_i = indices[indptr[i]:indptr[i+1]]

        The length of this subarray is given by counts[i].

        """
        s = self.faces.ravel().sort()

        one = torch.tensor([1], device=self.device())
        indptr = s.values.diff(prepend=one, append=one).nonzero().squeeze(-1)
        counts = s.values.bincount()

        # face indices
        indices = torch.arange(self.n_faces, device=self.device()).repeat_interleave(
            self.vertices_per_face
        )[s.indices]

        return indices, indptr, counts

    @staticmethod
    def make_edge_hash(edges, n):
        return n * edges[:, 0] + edges[:, 1]

    @staticmethod
    def undo_edge_hash(edge_hash, n):
        # - the second part of the hash is like a decimal, hence floor division
        # removes that part leavning only the first part
        # - modulus removes the contribution of the first part but leaves the
        # second part intact as the maximum value is n-1.
        return torch.stack((edge_hash // n, edge_hash % n), dim=1)

    def n_faces_to_n_vertices(self):
        return self.n_faces // 2 + 2

    # def get_n_vertices_for_lower_level(self):
    #     return self.n_faces_to_n_vertices(self.n_faces // self.subdivision_factor)

    def n_vertices_lower_order(self):
        return (self.n_vertices + 6) // 4

    def set_topology_information(self, retain_edge_order=True):
        """Make (sparse) adjacency matrix of vertices with connections as specified
        by `el`.

        `edges` includes each edge twice - once in each direction - that is, the
        edge between v0 and v1 is included as (v0,v1) and (v1,v0).

        PARAMETERS
        ----------
        el : ndarray
            n x m array describing the connectivity of the elements (e.g.,
            n x 3 for a triangulated of surface).
        with_diag : bool
            Include ones on the diagonal (default = False).

        RETURNS
        -------
        edges : torch.Tensor
            shape (# edges, 2)
        """
        vs = torch.arange(self.vertices_per_face, device=self.device())
        vertex_opposite_edge = torch.cat(
            [
                vs[torch.isin(vs, e, assume_unique=True, invert=True)]
                for e in self.edge_pairs
            ]
        )
        # indices of edge pairs associated with vertex_opposite_edge
        vertex_edges = torch.stack(
            [torch.where(v == self.edge_pairs)[0] for v in vertex_opposite_edge]
        )

        # Vertex adjacency and face-to-edge mapping
        # -----------------------------------------
        # no need to sort as unique will do that anyway
        edges = self.get_edges_ravelled(
            sort_within=True
        )  # e.g., (1,0) and (0,1) -> (0,1)

        # trick from pytorch3d: use a hash to speed up the call to unique which
        # is otherwise slow
        # ensure int64 to avoid overflow (assumes fewer than ~3e9 vertices)
        h = self.make_edge_hash(edges.to(torch.int64), self.n_vertices)

        if retain_edge_order:
            # remap the hashed edges to 0, ..., n_edges-1 preserving original
            # order of the edges
            x = torch.zeros(h.shape, dtype=self.dtype, device=self.device())
            hb = h.argsort(stable=True).to(self.dtype)
            x[hb] = hb[::2].repeat_interleave(2)  # each edge occurs exactly twice
            u, idx = x.unique(return_inverse=True)
            u_edges = edges[u]
            u_edges_prev_order = u_edges[u_edges[:, 0] < self.n_vertices_lower_order()]
        else:
            # this sorts the edges based on the hash
            u, idx = h.unique(return_inverse=True)
            u_edges = self.undo_edge_hash(u, self.n_vertices).to(self.dtype)
            # this eliminates all edges between upsampled vertices
            u_edges_prev_order = u_edges[: len(u) // 2]

        # back to dtype of faces
        faces_to_edges = idx.to(self.dtype).reshape(self.n_faces, 3)
        u_edges = u_edges.to(self.dtype)
        u_edges_prev_order = u_edges_prev_order.to(self.dtype)

        # if retain_edge_order:
        #     ind_sorted = idx.argsort(stable=True)
        #     # each edge occurs twice so we do not need the counts
        #     cum_sum = torch.arange(
        #         0, len(edges), 2, dtype=self.dtype, device=self.device()
        #     )
        #     # the more general solution
        #     # cum_sum = counts.cumsum(0)
        #     # cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))

        #     u_orig_order = u[ind_sorted[cum_sum].argsort()]

        #     # vertex_adjacency = self.undo_edge_hash(u_orig_order, self.n_vertices).to(
        #     #     self.dtype
        #     # )

        #     mapper = torch.zeros(h.amax() + 1, dtype=self.dtype)
        #     mapper[u_orig_order] = torch.arange(len(edges) / 2, dtype=self.dtype, device=self.device())
        #     faces_to_edges = mapper[h].reshape(self.n_faces, 3)

        # else:
        #     vertex_adjacency = self.undo_edge_hash(u, self.n_vertices).to(self.dtype)
        #     faces_to_edges = idx.reshape(-1, 3).to(self.dtype)

        # Face adjacency
        # --------------
        # Now sort the vertex-vertex edges
        # first column has 1st priority
        a0 = edges[:, 0].argsort()
        s0 = edges[a0]
        # second column has 2nd priority (actually, we just want to sub-sort `s0`)
        a1 = s0[:, 1].argsort()
        s1 = s0[a1]
        # stable=True keeps order of like items, hence both columns will be
        # sorted after this operation
        a2 = s1[:, 0].argsort(stable=True)
        # s2 = s1[a2] # the sorted edges

        faces_enum = (
            torch.arange(self.n_faces, dtype=self.dtype, device=self.device())[:, None]
            .expand_as(self.faces)
            .ravel()
        )
        face_adjacency = faces_enum[a0[a1[a2]]].reshape(-1, 2)

        self.register_buffer("vertex_edges", vertex_edges, persistent=False)
        self.register_buffer(
            "vertex_opposite_edge", vertex_opposite_edge, persistent=False
        )

        self.register_buffer("vertex_adjacency", u_edges, persistent=False)
        self.register_buffer("unique_edges", u_edges, persistent=False)
        self.register_buffer("face_adjacency", face_adjacency, persistent=False)
        self.register_buffer("faces_to_edges", faces_to_edges, persistent=False)
        self.register_buffer(
            "pool_index_reduce", u_edges_prev_order[:, 0], persistent=False
        )
        self.register_buffer(
            "pool_index_gather", u_edges_prev_order[:, 1], persistent=False
        )

        # Either array can be used as reduce or gather

        # Usage:
        #   B = batch size
        #   N = # vertices
        #   C = # channels
        #   r = torch.randn((B, N, C))
        #   x = torch.zeros((B, N, C))
        #   x.index_reduce_(
        #       1, reduce_index, gather_index, reduce="mean", include_self=True
        #   )
        self.register_buffer(
            "conv_index_reduce", self.vertex_adjacency.ravel(), persistent=False
        )
        self.register_buffer(
            "conv_index_gather", self.vertex_adjacency.flip(1).ravel(), persistent=False
        )

    def subdivide_faces(self):
        raise NotImplementedError("This method should be defined in a subclass.")

    def subdivide_vertices(self, vertices: torch.Tensor):
        """From V0-V1-V2, insert v3, v4, v5, at the midpoint of the edges
        (V0, V1), (V0, V2), (V1, V2).
        """
        assert (
            self.n_vertices == vertices.shape[1]
        ), f"Expected {self.n_vertices} but got {vertices.shape[1]}"
        return torch.cat([vertices, vertices[:, self.vertex_adjacency].mean(2)], dim=1)

    @classmethod
    def recursive_subdivision(cls, n: int, **kwargs):
        assert n >= 0

        topology = cls(**kwargs)
        topologies = torch.nn.ModuleList([topology])
        if "faces" in kwargs:
            del kwargs["faces"]
        for _ in range(0, n):
            topology = cls(topology.subdivide_faces(), **kwargs)
            topologies.append(topology)

        return topologies


class DeepSurferTopology(Topology):
    def __init__(
        self,
        faces: torch.Tensor | None = None,
        edge_pairs: str | torch.Tensor | None = None,
    ):
        """Topology from the DeepSurfer package. This is defined for the left
        hemisphere. Hence, to be valid for right hemisphere, we need to reverse
        the vertex order."""
        faces = deepsurfer_faces if faces is None else faces

        edge_pairs = torch.tensor(
            [[0, 1], [1, 2], [2, 0]], dtype=torch.int, device=faces.device
        )
        super().__init__(faces, edge_pairs, retain_edge_order=False)

    def subdivide_faces(self):
        """Subdivide all faces, increasing the face count by a factor of four.

        References
        ----------
        This is based on pytorch3d.ops.subdivide_meshes.
        """
        # The subdivision scheme
        #
        #               V0
        #              /  \
        #             /    \
        #            /      \
        #           /        \
        #          /    f0    \
        #         /            \
        #        v12 ---------- v13
        #       /  \          /  \
        #      /    \   f3   /    \
        #     /      \      /      \
        #    /   f2   \    /   f1   \
        #   /          \  /          \
        # V1 ---------- v14---------- V2

        # These are the faces made up entirely of new vertices)
        f_new = self.faces_to_edges + self.n_vertices

        # Concatenate each "original" vertex with the vertices placed on it's
        # adjacent edges such that the orientation (e.g., counter-clockwise) is
        # preserved
        f0 = torch.stack(
            [
                f_new[:, 0],  # edge 0-1
                f_new[:, 2],  # edge 2-0
                self.faces[:, 0],
            ],
            dim=1,
        )
        f1 = torch.stack(
            [
                f_new[:, 1],  # edge 1-2
                f_new[:, 0],  # edge 0-1
                self.faces[:, 1],
            ],
            dim=1,
        )
        f2 = torch.stack(
            [
                f_new[:, 2],  # 2
                f_new[:, 1],  # 1
                self.faces[:, 2],
            ],
            dim=1,
        )

        return torch.cat((f_new, f0, f1, f2))


# NOTE In order to reproduce the faces from
#   FS_DIR/average/surf/lh.sphere.ico{i}.reg
# we would need the following subdivision. However, when indexing into the
# vertices of fsaverage7 in order to obtain the lower resolution icos (e.g.,
# v[:N_FACES_ICO_0], v[:N_FACES_ICO_1], etc.), we simply have to reorder the
# faces each time!
#
# def fsaverage_reorder_faces_once(faces):
#     return torch.cat((faces[::4], faces.reshape(-1, 4, 3)[:, 1:].reshape(-1, 3)))
#
# def get_fsaverage_topology(ico_order: int = 0):
#     assert ico_order >= 0
#     topologies = [topology := FsAverageTopology()]
#     topologies_reordered = [topology_reordered := FsAverageTopology()]
#     for i in range(0, ico_order):
#         if i <= 3:
#             topology = FsAverageTopology(topology.subdivide_faces())
#             topologies.append(topology)
#         if ico_order >= 5:
#             topology_reordered = FsAverageTopology(fsaverage_reorder_faces_once(topology_reordered.subdivide_faces()))
#             topologies_reordered.append(topology_reordered)
#     return topologies + topologies_reordered[5:] if ico_order >= 5 else topologies


class FsAverageTopology(Topology):
    def __init__(
        self,
        faces: torch.Tensor | None = None,
        edge_pairs: str | torch.Tensor | None = None,
    ):
        faces = fsaverage_faces if faces is None else faces

        edge_pairs = torch.tensor(
            [[2, 0], [1, 2], [0, 1]], dtype=torch.int, device=faces.device
        )
        super().__init__(faces, edge_pairs, retain_edge_order=True)

    @staticmethod
    def reorder_subdivided_faces(faces):
        return torch.cat((faces[::4], faces.reshape(-1, 4, 3)[:, 1:].reshape(-1, 3)))

    def subdivide_faces(self):
        """Subdivide all faces, increasing the face count by a factor of four.

        References
        ----------
        This is based on pytorch3d.ops.subdivide_meshes.
        """
        # FreeSurfer subdivision scheme
        #
        #               V0------------ v15---------- V5
        #              /  \           / \           /
        #             /    \   f4    /   \   f5    /
        #            /      \       /     \       /
        #           /        \     /       \     /
        #          /    f0    \   /    f6   \   /
        #         /            \ /           \ /
        #        v14---------- v12 --------- v16
        #       /  \          /  \          /
        #      /    \   f3   /    \   f7   /
        #     /      \      /      \      /
        #    /   f3   \    /   f1   \    /
        #   /          \  /          \  /
        # V3 ---------- v13---------- V4
        #
        # Originally,
        #   F0 = (V0, V3, V4).
        #   F1 = (V0, V4, V5).
        # After subdivision, F0 and F1 are replaced by
        #   (f0, f1, f2, f3)
        #   (f4, f5, f6, f7)
        # i.e., vertices are placed on edges in this order: [2,0], [1,2], [0,1]

        new_vertices = torch.arange(
            self.n_vertices,
            4 * self.n_faces,
            dtype=self.faces.dtype,
            device=self.device(),
        )

        # the vertex to be inserted at each edge
        ve = new_vertices[self.faces_to_edges]

        faces = torch.stack(
            (
                torch.stack((self.faces[:, 0], ve[:, 2], ve[:, 0]), dim=1),
                torch.stack((ve[:, 0], ve[:, 1], self.faces[:, 2]), dim=1),
                torch.stack((ve[:, 2], ve[:, 1], ve[:, 0]), dim=1),
                torch.stack((ve[:, 2], self.faces[:, 1], ve[:, 1]), dim=1),
            ),
            dim=1,
        ).reshape(4 * self.n_faces, 3)

        return self.reorder_subdivided_faces(faces)


class CerebellumTopology(DeepSurferTopology):
    def __init__(
        self,
        faces: torch.Tensor | None = None,
        edge_pairs: str | torch.Tensor | None = None,
    ):
        faces = cerebellum_faces if faces is None else faces
        super().__init__(faces, edge_pairs)


class NonManifoldTopology(Topology):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def n_faces_to_n_vertices(self):
        return self.faces.unique().size().numel()

    def set_topology_information(self, retain_edge_order=True):
        """Make (sparse) adjacency matrix of vertices with connections as specified
        by `el`.

        `edges` includes each edge twice - once in each direction - that is, the
        edge between v0 and v1 is included as (v0,v1) and (v1,v0).

        PARAMETERS
        ----------
        el : ndarray
            n x m array describing the connectivity of the elements (e.g.,
            n x 3 for a triangulated of surface).
        with_diag : bool
            Include ones on the diagonal (default = False).

        RETURNS
        -------
        edges : torch.Tensor
            shape (# edges, 2)
        """
        vs = torch.arange(self.vertices_per_face, device=self.device())
        vertex_opposite_edge = torch.cat(
            [
                vs[torch.isin(vs, e, assume_unique=True, invert=True)]
                for e in self.edge_pairs
            ]
        )
        # indices of edge pairs associated with vertex_opposite_edge
        vertex_edges = torch.stack(
            [torch.where(v == self.edge_pairs)[0] for v in vertex_opposite_edge]
        )

        # Vertex adjacency and face-to-edge mapping
        # -----------------------------------------
        # no need to sort as unique will do that anyway, e.g., (1,0) and (0,1) -> (0,1)
        edges = self.get_edges_ravelled(sort_within=True)

        # trick from pytorch3d: use a hash to speed up the call to unique which
        # is otherwise slow
        # ensure int64 to avoid overflow (assumes fewer than ~3e9 vertices)
        h = self.make_edge_hash(edges.to(torch.int64), self.n_vertices)

        if retain_edge_order:
            # remap the hashed edges to 0, ..., n_edges-1 preserving original
            # order of the edges
            x = torch.zeros(h.shape, dtype=self.dtype, device=self.device())
            hb = h.argsort(stable=True).to(self.dtype)
            _, c = h[hb].unique_consecutive(return_counts=True)
            uhb = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int, device=self.device()),
                    c.cumsum(0)[:-1].to(torch.int),
                ]
            )
            x[hb] = hb[uhb.repeat_interleave(c)]
            u, idx, e_counts = x.unique(return_inverse=True, return_counts=True)
            u_edges = edges[u]
            u_edges_prev_order = u_edges[u_edges[:, 0] < self.n_vertices_lower_order()]
        else:
            # this sorts the edges based on the hash
            u, idx = h.unique(return_inverse=True)
            u_edges = self.undo_edge_hash(u, self.n_vertices).to(self.dtype)
            # this eliminates all edges between upsampled vertices
            u_edges_prev_order = u_edges[: len(u) // 2]

        # back to dtype of faces
        faces_to_edges = idx.to(self.dtype).reshape(self.n_faces, 3)
        u_edges = u_edges.to(self.dtype)
        u_edges_prev_order = u_edges_prev_order.to(self.dtype)

        # Face adjacency
        # --------------
        s = faces_to_edges.ravel().sort()
        _, i, c = s.values.unique_consecutive(return_inverse=True, return_counts=True)
        # only edges which occur twice contribute to face adjacency
        assert c.amin() >= 1 and c.max() == 2
        faces_enum = s.indices[c[i] == 2] // 3
        face_adjacency = faces_enum.reshape(-1, 2)

        # vertices which are part of a non-manifold edge
        nmv = u_edges[e_counts == 1].unique()
        self.register_buffer("non_manifold_vertices", nmv, persistent=False)
        # vertices which are NOT part of a non-manifold edge
        tmp = torch.ones(self.n_vertices, dtype=torch.bool, device=self.device())
        tmp[nmv] = False
        mv = tmp.nonzero().squeeze(1)
        self.register_buffer("manifold_vertices", mv, persistent=False)

        # faces which include a non-manifold edge
        nmf = torch.unique(s.indices[c[i] == 1] // 3)
        self.register_buffer("non_manifold_faces", nmf, persistent=False)
        # faces which DOES NOT include a non-manifold edge
        tmp = torch.ones(self.n_faces, dtype=torch.bool, device=self.device())
        tmp[nmf] = False
        mf = tmp.nonzero().squeeze(1)
        self.register_buffer("manifold_faces", mf, persistent=False)

        self.register_buffer("vertex_edges", vertex_edges, persistent=False)
        self.register_buffer(
            "vertex_opposite_edge", vertex_opposite_edge, persistent=False
        )

        self.register_buffer("vertex_adjacency", u_edges, persistent=False)
        self.register_buffer("unique_edges", u_edges, persistent=False)
        self.register_buffer("face_adjacency", face_adjacency, persistent=False)
        self.register_buffer("faces_to_edges", faces_to_edges, persistent=False)
        self.register_buffer(
            "pool_index_reduce", u_edges_prev_order[:, 0], persistent=False
        )
        self.register_buffer(
            "pool_index_gather", u_edges_prev_order[:, 1], persistent=False
        )

        # Either array can be used as reduce or gather

        # Usage:
        #   B = batch size
        #   N = # vertices
        #   C = # channels
        #   r = torch.randn((B, N, C))
        #   x = torch.zeros((B, N, C))
        #   x.index_reduce_(
        #       1, reduce_index, gather_index, reduce="mean", include_self=True
        #   )
        self.register_buffer(
            "conv_index_reduce", self.vertex_adjacency.ravel(), persistent=False
        )
        self.register_buffer(
            "conv_index_gather", self.vertex_adjacency.flip(1).ravel(), persistent=False
        )


deepsurfer_faces = torch.tensor(
    [
        [0, 2, 1],
        [0, 3, 2],
        [3, 4, 2],
        [5, 1, 2],
        [6, 5, 2],
        [7, 5, 6],
        [7, 6, 8],
        [6, 9, 8],
        [6, 2, 10],
        [6, 10, 9],
        [10, 2, 11],
        [12, 7, 8],
        [12, 13, 7],
        [14, 1, 5],
        [7, 14, 5],
        [13, 14, 7],
        [14, 15, 1],
        [16, 0, 1],
        [15, 16, 1],
        [17, 15, 14],
        [18, 16, 15],
        [17, 19, 15],
        [18, 15, 19],
        [20, 19, 17],
        [21, 17, 14],
        [22, 18, 19],
        [22, 19, 20],
        [23, 18, 22],
        [18, 24, 16],
        [24, 18, 23],
        [16, 25, 0],
        [25, 16, 24],
        [26, 3, 0],
        [3, 26, 4],
        [26, 0, 25],
        [27, 25, 24],
        [4, 26, 27],
        [27, 26, 25],
        [28, 24, 23],
        [27, 24, 28],
        [4, 27, 29],
        [29, 27, 28],
        [2, 4, 30],
        [30, 4, 29],
        [2, 30, 31],
        [11, 2, 32],
        [32, 2, 31],
        [33, 32, 31],
        [33, 31, 34],
        [31, 30, 35],
        [31, 35, 34],
        [23, 36, 28],
        [37, 30, 29],
        [30, 37, 35],
        [38, 29, 28],
        [37, 29, 38],
        [39, 28, 36],
        [38, 28, 39],
        [39, 36, 40],
        [13, 12, 41],
        [42, 38, 39],
        [39, 40, 42],
        [40, 41, 42],
        [43, 33, 34],
        [44, 14, 13],
        [44, 21, 14],
        [41, 44, 13],
        [41, 40, 44],
        [45, 20, 17],
        [45, 17, 21],
        [22, 20, 45],
        [23, 22, 46],
        [46, 36, 23],
        [45, 46, 22],
        [36, 47, 40],
        [47, 21, 44],
        [40, 47, 44],
        [47, 45, 21],
        [46, 47, 36],
        [47, 46, 45],
        [48, 37, 38],
        [48, 38, 42],
        [41, 12, 49],
        [49, 42, 41],
        [12, 8, 50],
        [49, 12, 50],
        [34, 35, 51],
        [35, 37, 51],
        [51, 37, 48],
        [52, 48, 42],
        [52, 42, 49],
        [50, 52, 49],
        [51, 48, 52],
        [53, 52, 50],
        [51, 52, 53],
        [43, 34, 54],
        [54, 34, 51],
        [54, 51, 53],
        [8, 9, 55],
        [8, 55, 50],
        [50, 55, 53],
        [55, 54, 53],
        [56, 43, 54],
        [56, 54, 55],
        [55, 9, 57],
        [57, 43, 56],
        [57, 56, 55],
        [58, 33, 43],
        [58, 43, 57],
        [11, 32, 59],
        [59, 32, 33],
        [58, 59, 33],
        [9, 10, 60],
        [57, 9, 60],
        [57, 60, 58],
        [60, 59, 58],
        [10, 11, 61],
        [11, 59, 61],
        [10, 61, 60],
        [60, 61, 59],
    ],
    dtype=torch.int,
)

fsaverage_faces = torch.tensor(
    [
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 1],
        [0, 1, 2],
        [0, 2, 3],
        [3, 2, 8],
        [3, 8, 9],
        [3, 9, 4],
        [4, 9, 10],
        [4, 10, 5],
        [5, 10, 6],
        [5, 6, 1],
        [1, 6, 7],
        [1, 7, 2],
        [2, 7, 8],
        [8, 11, 9],
        [9, 11, 10],
        [10, 11, 6],
        [6, 11, 7],
        [7, 11, 8],
    ],
    dtype=torch.int,
)


cerebellum_faces = torch.tensor(
    [
        [96, 45, 23],
        [101, 79, 127],
        [128, 151, 3],
        [12, 243, 186],
        [20, 25, 17],
        [75, 42, 37],
        [62, 31, 146],
        [74, 13, 53],
        [58, 5, 80],
        [179, 35, 120],
        [245, 44, 35],
        [5, 51, 16],
        [15, 16, 56],
        [126, 69, 28],
        [19, 167, 18],
        [249, 239, 12],
        [151, 106, 3],
        [151, 50, 78],
        [169, 84, 42],
        [112, 70, 61],
        [88, 62, 146],
        [5, 16, 80],
        [50, 43, 78],
        [14, 119, 23],
        [38, 55, 7],
        [11, 112, 61],
        [11, 58, 65],
        [21, 249, 12],
        [58, 80, 178],
        [56, 137, 15],
        [52, 46, 91],
        [5, 58, 11],
        [70, 112, 4],
        [40, 82, 22],
        [20, 165, 71],
        [37, 74, 53],
        [76, 249, 21],
        [72, 70, 4],
        [69, 43, 0],
        [11, 63, 5],
        [102, 29, 14],
        [38, 18, 82],
        [242, 220, 113],
        [113, 38, 82],
        [41, 6, 135],
        [249, 76, 237],
        [33, 86, 30],
        [23, 92, 14],
        [247, 10, 77],
        [85, 3, 106],
        [36, 46, 39],
        [17, 243, 20],
        [13, 54, 53],
        [29, 93, 14],
        [63, 117, 51],
        [51, 5, 63],
        [34, 222, 190],
        [87, 78, 43],
        [7, 68, 19],
        [51, 56, 16],
        [71, 165, 57],
        [17, 100, 186],
        [57, 165, 247],
        [236, 86, 141],
        [48, 49, 39],
        [9, 39, 49],
        [73, 137, 56],
        [23, 79, 96],
        [155, 20, 71],
        [7, 55, 32],
        [33, 30, 6],
        [20, 155, 25],
        [46, 36, 91],
        [32, 245, 68],
        [33, 6, 8],
        [42, 75, 169],
        [34, 3, 85],
        [186, 100, 121],
        [2, 37, 42],
        [36, 8, 83],
        [96, 141, 9],
        [135, 104, 41],
        [61, 81, 63],
        [121, 26, 186],
        [69, 146, 28],
        [33, 36, 39],
        [30, 135, 6],
        [44, 245, 32],
        [60, 50, 139],
        [24, 90, 120],
        [82, 40, 113],
        [57, 90, 159],
        [62, 88, 127],
        [59, 26, 121],
        [128, 3, 130],
        [82, 137, 22],
        [119, 127, 79],
        [236, 141, 130],
        [7, 19, 38],
        [72, 1, 76],
        [9, 45, 96],
        [28, 31, 64],
        [8, 36, 33],
        [63, 11, 61],
        [190, 236, 130],
        [6, 41, 8],
        [78, 106, 151],
        [104, 103, 41],
        [39, 46, 48],
        [202, 58, 178],
        [89, 85, 106],
        [59, 21, 26],
        [131, 1, 4],
        [126, 163, 69],
        [36, 83, 91],
        [131, 114, 237],
        [105, 75, 37],
        [3, 34, 130],
        [106, 78, 89],
        [134, 157, 73],
        [80, 16, 15],
        [93, 119, 14],
        [126, 28, 64],
        [84, 29, 42],
        [2, 74, 37],
        [53, 105, 37],
        [65, 202, 248],
        [159, 71, 57],
        [68, 245, 154],
        [33, 9, 86],
        [144, 76, 21],
        [13, 49, 48],
        [122, 49, 13],
        [147, 126, 64],
        [117, 125, 51],
        [84, 138, 133],
        [97, 110, 70],
        [76, 1, 237],
        [170, 128, 130],
        [253, 150, 77],
        [171, 74, 2],
        [54, 48, 158],
        [53, 54, 158],
        [111, 143, 17],
        [155, 111, 25],
        [184, 181, 59],
        [58, 202, 65],
        [110, 61, 70],
        [118, 224, 222],
        [222, 34, 246],
        [125, 134, 51],
        [93, 62, 119],
        [133, 64, 31],
        [35, 154, 245],
        [87, 43, 69],
        [105, 136, 75],
        [34, 85, 246],
        [8, 41, 94],
        [15, 18, 80],
        [78, 87, 109],
        [19, 68, 153],
        [169, 75, 115],
        [42, 29, 102],
        [76, 144, 72],
        [93, 29, 84],
        [220, 225, 66],
        [94, 98, 83],
        [183, 145, 52],
        [153, 167, 19],
        [55, 220, 66],
        [110, 149, 81],
        [148, 47, 184],
        [228, 22, 95],
        [85, 124, 246],
        [96, 79, 238],
        [55, 132, 32],
        [85, 160, 124],
        [141, 170, 130],
        [187, 117, 149],
        [190, 222, 241],
        [104, 232, 142],
        [179, 247, 77],
        [51, 134, 56],
        [227, 135, 30],
        [178, 80, 18],
        [18, 15, 82],
        [163, 140, 87],
        [183, 91, 98],
        [145, 99, 52],
        [68, 154, 153],
        [2, 102, 171],
        [181, 144, 21],
        [72, 144, 97],
        [70, 72, 97],
        [81, 149, 117],
        [53, 158, 212],
        [160, 85, 89],
        [50, 60, 123],
        [65, 67, 112],
        [79, 23, 119],
        [144, 206, 97],
        [52, 158, 48],
        [151, 128, 139],
        [237, 1, 131],
        [64, 168, 147],
        [59, 121, 194],
        [104, 107, 103],
        [52, 91, 183],
        [8, 94, 83],
        [1, 72, 4],
        [132, 55, 66],
        [117, 63, 81],
        [95, 22, 73],
        [236, 190, 241],
        [126, 147, 204],
        [103, 107, 172],
        [136, 115, 75],
        [221, 179, 120],
        [101, 88, 123],
        [98, 152, 183],
        [90, 221, 120],
        [238, 79, 101],
        [48, 46, 52],
        [87, 140, 109],
        [126, 204, 163],
        [107, 104, 142],
        [118, 246, 174],
        [198, 192, 153],
        [31, 93, 133],
        [89, 78, 109],
        [13, 74, 122],
        [21, 59, 181],
        [156, 89, 109],
        [98, 94, 164],
        [73, 56, 134],
        [52, 99, 158],
        [108, 14, 92],
        [150, 182, 77],
        [115, 184, 162],
        [205, 24, 250],
        [87, 69, 163],
        [98, 91, 83],
        [138, 84, 169],
        [167, 178, 18],
        [133, 168, 64],
        [116, 198, 154],
        [154, 35, 116],
        [43, 50, 123],
        [71, 159, 155],
        [162, 138, 115],
        [159, 185, 155],
        [172, 107, 217],
        [44, 32, 132],
        [235, 124, 160],
        [84, 133, 93],
        [92, 45, 122],
        [179, 77, 182],
        [240, 188, 66],
        [112, 11, 65],
        [157, 134, 210],
        [116, 179, 182],
        [214, 136, 105],
        [123, 60, 101],
        [139, 50, 151],
        [188, 132, 66],
        [134, 125, 210],
        [251, 118, 174],
        [156, 160, 89],
        [166, 213, 150],
        [100, 161, 121],
        [218, 140, 211],
        [207, 194, 121],
        [44, 250, 24],
        [153, 154, 198],
        [149, 110, 177],
        [212, 105, 53],
        [105, 212, 214],
        [203, 161, 100],
        [17, 143, 100],
        [195, 172, 191],
        [164, 103, 172],
        [164, 94, 103],
        [145, 183, 216],
        [173, 142, 230],
        [156, 109, 140],
        [186, 243, 17],
        [98, 164, 152],
        [148, 115, 136],
        [81, 61, 110],
        [127, 88, 101],
        [132, 188, 250],
        [212, 158, 99],
        [110, 176, 177],
        [156, 200, 160],
        [60, 139, 238],
        [110, 97, 176],
        [157, 199, 95],
        [102, 2, 42],
        [92, 23, 45],
        [92, 171, 108],
        [34, 190, 130],
        [195, 164, 172],
        [99, 145, 197],
        [107, 142, 173],
        [197, 214, 99],
        [215, 188, 180],
        [174, 246, 124],
        [73, 157, 95],
        [111, 155, 175],
        [166, 193, 213],
        [133, 138, 168],
        [48, 54, 13],
        [90, 189, 159],
        [144, 181, 47],
        [136, 47, 148],
        [100, 143, 203],
        [175, 196, 111],
        [198, 116, 182],
        [68, 7, 32],
        [227, 232, 135],
        [199, 234, 95],
        [213, 182, 150],
        [17, 25, 111],
        [221, 57, 247],
        [211, 196, 175],
        [216, 183, 152],
        [205, 189, 24],
        [138, 162, 207],
        [204, 211, 163],
        [199, 217, 234],
        [251, 230, 244],
        [88, 146, 0],
        [192, 198, 213],
        [125, 117, 187],
        [214, 212, 99],
        [194, 184, 59],
        [192, 248, 167],
        [167, 153, 192],
        [233, 174, 235],
        [248, 193, 65],
        [208, 147, 168],
        [96, 238, 170],
        [149, 216, 187],
        [157, 191, 199],
        [214, 47, 136],
        [178, 167, 202],
        [167, 248, 202],
        [248, 192, 213],
        [97, 206, 176],
        [15, 137, 82],
        [195, 210, 187],
        [210, 195, 191],
        [107, 173, 223],
        [209, 200, 156],
        [156, 140, 218],
        [250, 44, 132],
        [213, 193, 248],
        [168, 138, 207],
        [189, 90, 24],
        [162, 194, 207],
        [67, 131, 4],
        [164, 195, 201],
        [184, 115, 148],
        [155, 185, 175],
        [206, 144, 47],
        [101, 60, 238],
        [152, 201, 216],
        [189, 185, 159],
        [253, 10, 229],
        [147, 203, 204],
        [103, 94, 41],
        [169, 115, 138],
        [10, 247, 165],
        [218, 209, 156],
        [38, 19, 18],
        [147, 208, 203],
        [177, 176, 197],
        [226, 219, 235],
        [211, 204, 196],
        [77, 10, 253],
        [201, 152, 164],
        [177, 216, 149],
        [219, 233, 235],
        [88, 0, 123],
        [121, 161, 208],
        [203, 143, 204],
        [213, 198, 182],
        [140, 163, 211],
        [209, 189, 205],
        [205, 226, 209],
        [74, 171, 122],
        [128, 170, 139],
        [217, 199, 191],
        [121, 208, 207],
        [113, 40, 242],
        [244, 118, 251],
        [170, 141, 96],
        [45, 9, 49],
        [211, 175, 218],
        [187, 201, 195],
        [205, 215, 219],
        [208, 161, 203],
        [218, 175, 185],
        [47, 214, 206],
        [216, 201, 187],
        [209, 185, 189],
        [168, 207, 208],
        [184, 47, 181],
        [214, 197, 206],
        [215, 180, 233],
        [102, 108, 171],
        [197, 145, 177],
        [38, 113, 55],
        [233, 219, 215],
        [184, 194, 162],
        [209, 218, 185],
        [206, 197, 176],
        [187, 210, 125],
        [191, 157, 210],
        [196, 143, 111],
        [204, 143, 196],
        [145, 216, 177],
        [123, 0, 43],
        [180, 188, 240],
        [223, 231, 234],
        [252, 223, 173],
        [242, 228, 231],
        [252, 225, 231],
        [49, 122, 45],
        [129, 252, 230],
        [224, 227, 241],
        [232, 230, 142],
        [95, 234, 228],
        [205, 250, 215],
        [230, 232, 244],
        [40, 228, 242],
        [244, 227, 224],
        [226, 235, 160],
        [160, 200, 226],
        [225, 242, 231],
        [129, 230, 251],
        [118, 244, 224],
        [129, 225, 252],
        [9, 33, 39],
        [230, 252, 173],
        [253, 239, 27],
        [139, 170, 238],
        [228, 234, 231],
        [122, 171, 92],
        [220, 55, 113],
        [236, 30, 86],
        [114, 27, 237],
        [222, 246, 118],
        [108, 102, 14],
        [124, 235, 174],
        [146, 31, 28],
        [241, 30, 236],
        [225, 220, 242],
        [180, 129, 251],
        [240, 129, 180],
        [217, 223, 234],
        [223, 252, 231],
        [179, 221, 247],
        [228, 40, 22],
        [226, 200, 209],
        [229, 239, 253],
        [112, 67, 4],
        [179, 116, 35],
        [131, 166, 114],
        [239, 229, 243],
        [237, 27, 249],
        [12, 26, 21],
        [249, 27, 239],
        [141, 86, 9],
        [222, 224, 241],
        [165, 20, 10],
        [35, 44, 120],
        [127, 119, 62],
        [31, 62, 93],
        [193, 166, 131],
        [217, 107, 223],
        [186, 26, 12],
        [129, 240, 225],
        [172, 217, 191],
        [137, 73, 22],
        [229, 20, 243],
        [57, 221, 90],
        [253, 27, 114],
        [225, 240, 66],
        [114, 166, 150],
        [44, 24, 120],
        [150, 253, 114],
        [174, 233, 251],
        [219, 226, 205],
        [250, 188, 215],
        [67, 65, 193],
        [251, 233, 180],
        [131, 67, 193],
        [232, 227, 244],
        [30, 241, 227],
        [104, 135, 232],
        [146, 69, 0],
        [243, 12, 239],
        [20, 229, 10],
    ],
    dtype=torch.int,
)
