import torch


# def get_recursively_subdivided_topology(n_recursions: int = 0, device=None):
#     assert n_recursions >= 0

#     topology = StandardTopology(device=device)
#     topologies = [topology]
#     for _ in range(0, n_recursions):
#         topology = StandardTopology(topology.subdivide_faces())
#         topologies.append(topology)

#     return topologies


# NOTE In order to reproduce the faces from
#   FS_DIR/average/surf/lh.sphere.ico{i}.reg
# we would need the following subdivision. However, when indexing into the
# vertices of fsaverage7 in order to obtain the lower resolution icos (e.g.,
# v[:N_FACES_ICO_0], v[:N_FACES_ICO_1], etc.), we simply have to reorder the
# faces each time!

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


# def get_fsaverage_topology(
#     ico_order: int = 0, device: str | torch.device | None = None
# ):
#     assert ico_order >= 0

#     topologies = [topology := FsAverageTopology(device=device)]
#     for _ in range(0, ico_order):
#         # reordering the faces changes the order in which vertices are added
#         # upon face subdivision!
#         topology = FsAverageTopology(
#             fsaverage_reorder_faces_once(topology.subdivide_faces()), device=device
#         )
#         topologies.append(topology)

#     return topologies


def fsaverage_reorder_faces_once(faces):
    return torch.cat((faces[::4], faces.reshape(-1, 4, 3)[:, 1:].reshape(-1, 3)))


# def fsaverage_reorder_faces(faces, order, n_ico0=20):

#     order1 = order + 1
#     shape = [n_ico0] + [4 for i in range(order)] + [3]
#     ff = faces.reshape(shape)

#     ffindexed = []
#     for i in range(order1):
#         indexer = [0 for _ in range(order1)]
#         for j in range(max(i, 1)):
#             indexer[j] = slice(None)
#         if i > 0:
#             indexer[i] = slice(1, None)

#         ffindexed.append(ff[*indexer].reshape(-1,3))

#     return torch.cat(ffindexed)


class Topology:
    def __init__(
        self,
        faces: torch.Tensor,
        edge_pairs: str | torch.Tensor | None = None,
        retain_edge_order: bool = True,
        device: str | torch.device | None = None,
    ):
        self.subdivision_factor = 4
        self.faces = faces
        match device:
            case str():
                self.device = torch.device(device)
            case None:
                self.device = faces.device
            case torch.device():
                self.device = device
            case _:
                raise ValueError()
        self.dtype = faces.dtype
        self.n_faces = faces.shape[0]
        self.vertices_per_face = faces.shape[1]
        self.n_vertices = self.n_faces_to_n_vertices(self.n_faces)
        self._reversed_face_order = (0, 2, 1)

        self.edge_pairs = (
            torch.tensor([[1, 2], [2, 0], [0, 1]], dtype=self.dtype, device=self.device)
            if edge_pairs is None
            else edge_pairs
        )

        self.set_topology_information(retain_edge_order)

    def reverse_face_orientation(self):
        self.faces = self.faces[:, self._reversed_face_order]
        # Ensure subdivide_faces is still valid
        self.faces_to_edges = self.faces_to_edges[:, self._reversed_face_order]

    def edges_from_faces(
        self,
        sort_dim0: bool = False,
        sort_dim1: bool = False,
    ):
        """Apply *within*-edge sort no matter what."""
        # pairs = torch.LongTensor([[0, 1], [1, 2], [2, 0]])
        # pairs = torch.tensor([[1, 2], [2, 0], [0, 1]], device=self.device)
        edges = torch.stack(
            [
                self.faces[:, self.edge_pairs[:, 0]],
                self.faces[:, self.edge_pairs[:, 1]],
            ],
            dim=2,
        )
        edges = edges.reshape((-1, 2))
        if sort_dim1:
            edges, _ = edges.sort(1)
        if sort_dim0:
            edge0, edge0_index = edges[:, 0].sort()
            edges[:, 0] = edge0
            edges[:, 1] = edges[edge0_index, 1]
        return edges

    def get_convolution_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """


        Parameters
        ----------
        a :
            Array on which the convolution is to be applied.
            shape = (batch size, # vertices, # channels)

        Returns
        -------
        reduce_index, gather_index : torch.Tensor, torch.Tensor
        """
        reduce_index = self.vertex_adjacency.ravel()
        gather_index = self.vertex_adjacency.flip(1).ravel()

        return reduce_index, gather_index

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

        own_features = self.subsample_array(features)
        pooled = torch.zeros_like(own_features).copy_(own_features)
        pooled.index_reduce_(
            -1,
            self.pool_index_reduce,
            features[..., self.pool_index_gather],
            reduce=reduce,
            include_self=True,
        )

        return pooled

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

        one = torch.tensor([1], device=self.device)
        indptr = s.values.diff(prepend=one, append=one).nonzero().squeeze(-1)
        counts = s.values.bincount()

        # face indices
        indices = torch.arange(self.n_faces, device=self.device).repeat_interleave(
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

    @staticmethod
    def n_faces_to_n_vertices(nf: int):
        return nf // 2 + 2

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

        # Vertex adjacency and face-to-edge mapping
        # -----------------------------------------
        # no need to sort as unique will do that anyway
        edges = self.edges_from_faces(sort_dim1=True)  # e.g., (1,0) and (0,1) -> (0,1)

        # trick from pytorch3d: use a hash to speed up the call to unique which
        # is otherwise slow
        # ensure int64 to avoid overflow (assumes fewer than ~3e9 vertices)
        h = self.make_edge_hash(edges.to(torch.int64), self.n_vertices)

        if retain_edge_order:
            # remap the hashed edges to 0, ..., n_edges-1 preserving original
            # order of the edges
            x = torch.zeros(h.shape, dtype=self.dtype, device=self.device)
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

        faces_to_edges = idx.reshape(self.n_faces, 3)

        # if retain_edge_order:
        #     ind_sorted = idx.argsort(stable=True)
        #     # each edge occurs twice so we do not need the counts
        #     cum_sum = torch.arange(
        #         0, len(edges), 2, dtype=self.dtype, device=self.device
        #     )
        #     # the more general solution
        #     # cum_sum = counts.cumsum(0)
        #     # cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))

        #     u_orig_order = u[ind_sorted[cum_sum].argsort()]

        #     # vertex_adjacency = self.undo_edge_hash(u_orig_order, self.n_vertices).to(
        #     #     self.dtype
        #     # )

        #     mapper = torch.zeros(h.amax() + 1, dtype=self.dtype)
        #     mapper[u_orig_order] = torch.arange(len(edges) / 2, dtype=self.dtype, device=self.device)
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
            torch.arange(self.n_faces, dtype=self.dtype, device=self.device)[:, None]
            .expand_as(self.faces)
            .ravel()
        )
        face_adjacency = faces_enum[a0[a1[a2]]].reshape(-1, 2)

        self.vertex_adjacency = self.unique_edges = u_edges
        self.face_adjacency = face_adjacency
        self.faces_to_edges = faces_to_edges

        self.pool_index_reduce = u_edges_prev_order[:, 0]
        self.pool_index_gather = u_edges_prev_order[:, 1]

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
        self.conv_index_reduce = self.vertex_adjacency.ravel()
        self.conv_index_gather = self.vertex_adjacency.flip(1).ravel()

    def subdivide_faces(self):
        raise NotImplementedError("This method should be defined in a subclass.")

    def subdivide_vertices(self, vertices):
        """From V0-V1-V2, insert v3, v4, v5, at the midpoint of the edges
        (V0, V1), (V0, V2), (V1, V2).
        """
        assert self.n_vertices == vertices.shape[-1]

        return torch.cat(
            [vertices, vertices[..., self.vertex_adjacency].mean(-1)], dim=-1
        )

    @classmethod
    def recursive_subdivision(cls, n: int, **kwargs):
        assert n >= 0

        topology = cls(**kwargs)
        topologies = [topology]
        if "faces" in kwargs:
            del kwargs["faces"]
        for _ in range(0, n):
            topology = cls(topology.subdivide_faces(), **kwargs)
            topologies.append(topology)

        return topologies


class StandardTopology(Topology):
    def __init__(
        self,
        faces: torch.Tensor | None = None,
        edge_pairs: str | torch.Tensor | None = None,
        device: str | torch.device | None = None,
    ):
        device = torch.device(device) if isinstance(device, str) else device

        faces = (
            torch.tensor(
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
                device=device,
            )
            if faces is None
            else faces
        )

        edge_pairs = torch.tensor(
            [[1, 2], [2, 0], [0, 1]], dtype=torch.int, device=device
        )
        super().__init__(faces, edge_pairs, retain_edge_order=False, device=device)

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
        #
        # We have
        #
        #   faces_to_edges[:, 0] = edge opposite faces[:, 0]
        #   faces_to_edges[:, 1] = edge opposite faces[:, 1]
        #   faces_to_edges[:, 2] = edge opposite faces[:, 2]
        #
        # i.e.,
        #
        #   edges = e12, e20, e01 = (v1,v2), (v2,v0), (v0,v1)

        # Concatenate each "original" vertex with the vertices placed on it's
        # adjacent edges such that the orientation (e.g., counter-clockwise) is
        # preserved. Please refer to the figure above
        new_vertex_indices = self.faces_to_edges + self.n_vertices

        f0 = torch.stack(
            [
                self.faces[:, 0],
                new_vertex_indices[:, self.edge_pairs[0, 1]],
                new_vertex_indices[:, self.edge_pairs[0, 0]],
            ],
            dim=1,
        )
        f1 = torch.stack(
            [
                self.faces[:, 1],
                new_vertex_indices[:, self.edge_pairs[1, 1]],
                new_vertex_indices[:, self.edge_pairs[1, 0]],
            ],
            dim=1,
        )
        f2 = torch.stack(
            [
                self.faces[:, 2],
                new_vertex_indices[:, self.edge_pairs[2, 1]],
                new_vertex_indices[:, self.edge_pairs[2, 0]],
            ],
            dim=1,
        )
        # Finally, the faces made up entirely of new vertices
        f3 = new_vertex_indices

        return torch.cat((f0, f1, f2, f3))


class FsAverageTopology(Topology):
    def __init__(
        self,
        faces: torch.Tensor | None = None,
        edge_pairs: str | torch.Tensor | None = None,
        device: str | torch.device | None = None,
    ):
        device = torch.device(device) if isinstance(device, str) else device

        faces = (
            torch.tensor(
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
                device=device,
            )
            if faces is None
            else faces
        )

        edge_pairs = torch.tensor(
            [[2, 0], [1, 2], [0, 1]], dtype=torch.int, device=device
        )
        super().__init__(faces, edge_pairs, retain_edge_order=True, device=device)

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
            device=self.device,
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