import torch

from brainnet.mesh.surface import TemplateSurfaces
from brainnet import sphere_utils

from brainnet.modules.losses import L1Loss, MSELoss, MSNELoss, MSCosSimLoss, SemiHardSNELoss

from brainnet.utils import unsqueeze_and_expand

# DECORATORS


def wrap_semi_symmetric(cls):
    """Decorator to compute symmetric loss of inputs."""

    class SemiSymmetric(cls):
        def __init__(
            self,
            sym_weights: list[float] | tuple[float, float] = (0.5, 0.5),
            *args,
            **kwargs,
        ) -> None:
            """Compute semi (weighted) symmetric loss of inputs.

            Parameters
            ----------
            sym_weights : list[float] | tuple
                weights[0] is the weight given to the loss from `y_pred` to the
                corresponding value in `y_true`, i.e,
                    sym_weights[0] * loss(y_pred, y_true[i_pred]) +
                    sym_weights[1] * loss(y_true, y_pred[i_true])

            """
            super().__init__(*args, **kwargs)
            assert len(sym_weights) == 2 and sum(sym_weights) == 1.0
            self.w = sym_weights

        def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            i_pred: torch.IntTensor,
            i_true: torch.IntTensor,
            w_pred: None | torch.Tensor = None,
            w_true: None | torch.Tensor = None,
        ):
            """
            Parameters
            ----------

            i_pred :
                Contains indices into `y_true`, i.e., y_true[i_pred[i]] is the
                item in y_true that corresponds to y_pred[i].
            i_true :
                Defined similarly to `i_pred` but contains indices into
                `y_pred`.

            Returns
            -------

            """
            i_pred = unsqueeze_and_expand(i_pred, y_true)
            i_true = unsqueeze_and_expand(i_true, y_pred)

            pt = super().forward(y_pred, y_true.gather(1, i_pred), w_pred)
            tp = super().forward(y_pred.gather(1, i_true), y_true, w_true)

            return self.w[0] * pt + self.w[1] * tp

            # if (batch_size := y_pred.shape[0]) > 1:
            #     batch_index = torch.arange(batch_size)[:, None]
            #     return self.w[0] * super().forward(
            #         y_pred, y_true[batch_index, i_pred], w_pred
            #     ) + self.w[1] * super().forward(
            #         y_pred[batch_index, i_true], y_true, w_true
            #     )

            # else:
            #     p = y_pred.squeeze(0)
            #     t = y_true.squeeze(0)
            #     w_pred = w_pred.squeeze(0) if w_pred is not None else w_pred
            #     w_true = w_true.squeeze(0) if w_true is not None else w_true
            #     return self.w[0] * super().forward(
            #         p, t[i_pred.squeeze(0)], w_pred
            #     ) + self.w[1] * super().forward(p[i_true.squeeze(0)], t, w_true)

    return SemiSymmetric


def wrap_index_matched_data(cls):
    class IndexMatchedData(cls):
        def __init__(
            self,
            value_key: str = "points",
            index_key: str = "chamfer_index",
            weight_key: str | None = None,
            *args,
            **kwargs,
        ) -> None:
            """Compute loss using (precomputed) data stored in the surface
            instance.
            """
            super().__init__(*args, **kwargs)
            self.value_key = value_key
            self.index_key = index_key
            self.weight_key = weight_key

        def forward(
            self,
            y_pred: TemplateSurfaces,
            y_true: TemplateSurfaces,
        ):
            if self.value_key == "points":
                p = y_pred.interpolated["points"]
                t = y_true.interpolated["points"]
            else:
                p = y_pred.interpolated["data"][self.value_key]
                t = y_true.interpolated["data"][self.value_key]
            if self.weight_key is None:
                return super().forward(
                    p,
                    t,
                    y_pred.interpolated["data"][self.index_key],
                    y_true.interpolated["data"][self.index_key],
                )
            else:
                return super().forward(
                    p,
                    t,
                    y_pred.interpolated["data"][self.index_key],
                    y_true.interpolated["data"][self.index_key],
                    y_pred.interpolated["data"][self.weight_key],
                    y_true.interpolated["data"][self.weight_key],
                )
    return IndexMatchedData


class MatchedDistanceLoss(MSNELoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred: TemplateSurfaces, y_true: TemplateSurfaces):
        """Average (squared) distance between matched vertices of surfaces a and b.

        Parameters
        ----------
        surface_a : TemplateSurfaces
            First surface
        surface_b : TemplateSurfaces
            Second surface.

        Returns
        -------
        loss
            _description_
        """
        weights = (
            y_true.vertex_data["weights"] if "weights" in y_true.vertex_data else None
        )
        return super().forward(y_pred.vertices, y_true.vertices, weights)


# class MatchedDistanceLoss2(MeanNormLoss):

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(self, y_pred: TemplateSurfaces, y_true: TemplateSurfaces):
#         """Average (squared) distance between matched vertices of surfaces a and b.

#         Parameters
#         ----------
#         surface_a : TemplateSurfaces
#             First surface
#         surface_b : TemplateSurfaces
#             Second surface.

#         Returns
#         -------
#         loss
#             _description_
#         """
#         weights = (
#             y_true.vertex_data["weights"] if "weights" in y_true.vertex_data else None
#         )
#         return super().forward(y_pred.vertices, y_true.vertices, weights)


# class SurfaceRMSELoss(RMSELoss):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(self, y_pred: TemplateSurfaces, y_true: TemplateSurfaces):
#         """Average (squared) distance between matched vertices of surfaces a and b.

#         Parameters
#         ----------
#         surface_a : TemplateSurfaces
#             First surface
#         surface_b : TemplateSurfaces
#             Second surface.

#         Returns
#         -------
#         loss
#             _description_
#         """
#         weights = (
#             y_true.vertex_data["weights"] if "weights" in y_true.vertex_data else None
#         )
#         return super().forward(y_pred.vertices, y_true.vertices, weights)


# class SurfaceEdgeRMSELoss(RMSELoss):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, y_pred, y_true):
#         # (n_batch, n_vertices, v_per_edge (= 2), coordinates)
#         edges = y_pred.vertices[:, y_pred.topology.vertex_adjacency]
#         y_pred_dist = edges.diff(dim=2).norm(dim=3)

#         edges = y_true.vertices[:, y_true.topology.vertex_adjacency]
#         y_true_dist = edges.diff(dim=2).norm(dim=3)

#         return super().forward(y_pred_dist, y_true_dist)


class CentralAngleLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        a_pred = sphere_utils.cart_to_sph(y_pred.vertices)[..., 1:]
        a_true = sphere_utils.cart_to_sph(y_true.vertices)[..., 1:]
        return sphere_utils.compute_central_angle(a_pred, a_true).mean()


class SphericalArcLoss(torch.nn.Module):
    def __init__(self, radius: float | None = None) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, y_pred, y_true):
        return (
            sphere_utils.compute_arc_length(
                y_pred.vertices, y_true.vertices, self.radius
            )
            .pow(2)
            .mean()
        )
        # return compute_arc_length(y_pred.vertices, y_true.vertices, self.radius).mean()


class SphericalAxisAlignedArcLoss_tensor(torch.nn.Module):
    def __init__(self, radius: float | None = None) -> None:
        """Compute the mean squared distance two points on the"""
        super().__init__()
        self.radius = radius

    def forward(self, y_pred, y_true):
        theta = (
            sphere_utils.compute_axis_aligned_arc_length(
                y_pred[..., 0], y_true[..., 0], self.radius
            )
            .pow(2)
            .mean()
            .sqrt()
        )
        phi = (
            sphere_utils.compute_axis_aligned_arc_length(
                y_pred[..., 1], y_true[..., 1], self.radius
            )
            .pow(2)
            .mean()
            .sqrt()
        )
        return theta + phi


class SphericalAxisAlignedArcLoss(torch.nn.Module):
    def __init__(self, radius: float | None = None) -> None:
        """Compute the mean squared distance two points on the"""
        super().__init__()
        self._tensor_loss = SphericalAxisAlignedArcLoss_tensor(radius)

    def forward(self, y_pred, y_true):
        return self._tensor_loss(y_pred.vertices, y_true.vertices)


# class SphericalEdgeLoss(RMSELoss):
#     def __init__(self, radius: float | None = None) -> None:
#         super().__init__()
#         self.saaa = SphericalAxisAlignedArcLoss_tensor(radius)

#     def forward(self, y_pred, y_true):
#         # (n_batch, n_vertices, v_per_edge (= 2), coordinates)
#         edge_vertices = y_pred.vertices[:, y_pred.topology.vertex_adjacency]
#         y_pred_dist = self.saaa(edge_vertices[:, :, 0], edge_vertices[:, :, 1])

#         edge_vertices = y_true.vertices[:, y_true.topology.vertex_adjacency]
#         y_true_dist = self.saaa(edge_vertices[:, :, 0], edge_vertices[:, :, 1])

#         return super().forward(y_pred_dist, y_true_dist)


class SphericalNormalLoss(MSNELoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred):
        # origin to barycenter
        bc = y_pred.compute_face_barycenters()
        bc = bc / bc.norm(dim=-1, keepdim=True)
        # bc = sph_to_cart(1.0, bc[..., 0], bc[..., 1]) # normalized

        # v_orig = y_pred.vertices.clone()
        # y_pred.vertices = sph_to_cart(1.0, y_pred.vertices[..., 0], y_pred.vertices[..., 1])
        n = y_pred.compute_face_normals()
        # y_pred.vertices = v_orig

        return super().forward(n, bc)


SemiSymmetricMSELoss = wrap_semi_symmetric(MSELoss)
SampledSemiSymmetricMSELoss = wrap_index_matched_data(SemiSymmetricMSELoss)

SemiSymmetricMSNELoss = wrap_semi_symmetric(MSNELoss)
SampledSemiSymmetricMSNormLoss = wrap_index_matched_data(SemiSymmetricMSNELoss)

SemiSymmetricL1Loss = wrap_semi_symmetric(L1Loss)
SampledSemiSymmetricL1Loss = wrap_index_matched_data(SemiSymmetricL1Loss)

SemiSymmetricCosSimLoss = wrap_semi_symmetric(MSCosSimLoss)
SampledSemiSymmetricCosSimLoss = wrap_index_matched_data(SemiSymmetricCosSimLoss)

# SemiHard
SemiSymmetricSemiHardNELoss = wrap_semi_symmetric(SemiHardSNELoss)
SampledSemiSymmetricSemiHardSNormLoss = wrap_index_matched_data(SemiSymmetricSemiHardNELoss)


# REGULARIZATION

class FaceNormalConsistencyLoss(MSNELoss): # MSCosSimLoss
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred):
        normals = y_pred.compute_face_normals()
        edge_face_normals = normals[:, y_pred.topology.face_adjacency]
        a, b = edge_face_normals.unbind(2)
        return super().forward(a, b)


class SmoothnessLoss(MSNELoss):
    def __init__(self, dim=-1) -> None:
        """Penalize smoothness of the surface as measured by the Laplacian."""
        super().__init__(dim)

    def forward(self, y_pred):
        ri, gi = y_pred.topology.get_convolution_indices()
        return super().forward(
            y_pred.vertices[y_pred.batch_ix, ri], y_pred.vertices[y_pred.batch_ix, gi]
        )


class SpringForceLoss(MSELoss):
    def __init__(self, *args, **kwargs) -> None:
        """Penalize differences in distances between neighboring vertices."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_distances(s):
        """Compute distances between a vertex and all its neighbors."""
        ri, gi = s.topology.get_convolution_indices()
        return torch.linalg.norm(
            s.vertices[s.batch_ix, ri] - s.vertices[s.batch_ix, gi], dim=-1
        )

    def forward(self, y_pred, y_true):
        dp = self.compute_distances(y_pred)
        dt = self.compute_distances(y_true)
        return super().forward(dp, dt)


class VertexToVertexAngleLoss(MSCosSimLoss):


    def __init__(self, dim: int = -1) -> None:
        """_summary_

        DEGREES     COSINE
        -------     ------
          0          1.000
         10          0.985
         20          0.940
         30          0.866
         40          0.766
         50          0.643
         60          0.500
         70          0.342
         80          0.174
         90          0.000
        100         -0.174
        110         -0.342
        120         -0.500
        130         -0.643
        140         -0.766
        150         -0.866
        160         -0.940
        170         -0.985
        180         -1.000

        Parameters
        ----------
        cutoff : _type_, optional
            Cutoff specified as cos(angle).
        dim : int, optional
            _description_, by default 2
        """
        super().__init__(dim=dim)

        self.inner = "white"
        self.outer = "pial"
        # self.cutoff = cutoff
        # self.cutoff = torch.deg2rad(cutoff_degrees).cos()

    def forward(self, y_pred: dict[str, TemplateSurfaces]):
        vw = y_pred[self.inner].vertices
        vp = y_pred[self.outer].vertices
        nw = y_pred[self.inner].compute_vertex_normals()
        return super().forward(vp-vw, nw)

        # cos = super().forward(vp-vw, nw)

        # if self.cutoff is not None:
        #     cos_valid = cos[cos <= self.cutoff]
        #     # avoid empty array
        #     loss = (1 - cos_valid)**2 if cos_valid.any() else torch.tensor([0.0], device=cos_valid.device)
        # else:
        #     loss = (1 - cos)**2

        # return loss.mean() # cos, loss,


class EdgeLengthVarianceLoss(MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, surfaces: TemplateSurfaces):
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
            Average variance over batch.
        """
        edge_length = surfaces.compute_edge_lengths()
        n = edge_length.shape[1]
        # new mean is 1
        edge_length_mu1 = edge_length * n / edge_length.sum(1)[:, None]
        # MSE to the mean
        return super().forward(edge_length_mu1, 1)


# OTHER

class SelfIntersectionCount(torch.nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, surface: TemplateSurfaces):
        if surface.n_batch == 1:
            _, n = surface.compute_self_intersections()
        else:
            n = torch.mean(
                torch.stack(
                    [i for _, i in surface.compute_self_intersections()]
                ).float()
            )
        return 100.0 * n / surface.topology.n_vertices if self.normalize else n


# class SymmetricThicknessLoss(SemiSymmetricMSNormLoss):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.inner = "white"
#         self.outer = "pial"

#     def compute_thickness_vector(self, y):
#         y_in = y[self.inner]
#         y_out = y[self.outer]
#         ref = y_in  # arbitrary

#         # these are indices into y_pred[h][outer]!
#         index = ref.nearest_neighbor_tensors(
#             y_in.vertex_data["points_sampled"],
#             y_out.vertex_data["points_sampled"],
#         )
#         ix = y_out.batch_ix[:, None]
#         y_inner_th_vec = (
#             y_out.vertex_data["points_sampled"][ix, index]
#             - y_in.vertex_data["points_sampled"]
#         )

#         # these are indices into y_pred[h][inner]!
#         index = ref.nearest_neighbor_tensors(
#             y_out.vertex_data["points_sampled"],
#             y_in.vertex_data["points_sampled"],
#         )
#         ix = y_in.batch_ix[:, None]
#         y_outer_th_vec = (
#             y_out.vertex_data["points_sampled"]
#             - y_in.vertex_data["points_sampled"][ix, index]
#         )

#         # inner_th_vec
#         #   vector from sampled points on INNER surface to closest
#         #   sampled point on OUTER surface
#         # outer_th_vec
#         #   vector from sampled points on OUTER surface to closest
#         #   sampled point on INNER surface

#         return y_inner_th_vec, y_outer_th_vec

#     def forward(self, y_pred, y_true):
#         """Compute the thickness vectors"""

#         yp_inner_th, yp_outer_th = self.compute_thickness_vector(y_pred)
#         yt_inner_th, yt_outer_th = self.compute_thickness_vector(y_true)

#         return 0.5 * super().forward(
#             yp_inner_th,
#             yt_inner_th,
#             y_pred[self.inner].vertex_data["sampled_index"],
#             y_true[self.inner].vertex_data["sampled_index"],
#         ) + 0.5 * super().forward(
#             yp_outer_th,
#             yt_outer_th,
#             y_pred[self.outer].vertex_data["sampled_index"],
#             y_true[self.outer].vertex_data["sampled_index"],
#         )