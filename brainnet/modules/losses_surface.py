import torch

from brainnet.mesh.surface import Surface
from brainnet import sphere_utils

from brainnet.modules.losses import (
    L1Loss,
    NL1Loss,
    NegLogLikLoss,
    NSELoss,
    MSELoss,
    MSNELoss,
    MSCosSimLoss,
    SemiHardSNELoss,
)

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
            x: torch.Tensor,
            y: torch.Tensor,
            xi: torch.IntTensor,
            yi: torch.IntTensor,
            xw: None | torch.Tensor = None,
            yw: None | torch.Tensor = None,
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
            xi = unsqueeze_and_expand(xi, y)
            yi = unsqueeze_and_expand(yi, x)

            pt = super().forward(x, y.gather(1, xi), xw)
            tp = super().forward(x.gather(1, yi), y, yw)

            return self.w[0] * pt + self.w[1] * tp

    return SemiSymmetric


def wrap_index_matched_data(cls):
    class IndexMatchedData(cls):
        def __init__(
            self,
            value_key: tuple[str, str] = ("interpolated", "points"),
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

        def forward(self, y_pred: Surface, y_true: Surface):
            match self.value_key:
                case ("mesh", "vertices"):
                    pv = y_pred.vertices
                    tv = y_true.vertices
                    pi = y_pred.vertex_data[self.index_key]
                    ti = y_true.vertex_data[self.index_key]
                    if self.weight_key is None:
                        pw = tw = None
                    else:
                        pw = y_pred.vertex_data[self.weight_key]
                        tw = y_true.vertex_data[self.weight_key]
                case ("mesh", _):
                    pv = y_pred.vertex_data[self.value_key[1]]
                    tv = y_true.vertex_data[self.value_key[1]]
                    pi = y_pred.vertex_data[self.index_key]
                    ti = y_true.vertex_data[self.index_key]
                    if self.weight_key is None:
                        pw = tw = None
                    else:
                        pw = y_pred.vertex_data[self.weight_key]
                        tw = y_true.vertex_data[self.weight_key]
                case ("interpolated", "points"):
                    pv = y_pred.interpolated["points"]
                    tv = y_true.interpolated["points"]
                    pi = y_pred.interpolated["data"][self.index_key]
                    ti = y_true.interpolated["data"][self.index_key]
                    if self.weight_key is None:
                        pw = tw = None
                    else:
                        pw = y_pred.interpolated["data"][self.weight_key]
                        tw = y_true.interpolated["data"][self.weight_key]
                case ("interpolated", _):
                    pv = y_pred.interpolated["data"][self.value_key[1]]
                    tv = y_true.interpolated["data"][self.value_key[1]]
                    pi = y_pred.interpolated["data"][self.index_key]
                    ti = y_true.interpolated["data"][self.index_key]
                    if self.weight_key is None:
                        pw = tw = None
                    else:
                        pw = y_pred.interpolated["data"][self.weight_key]
                        tw = y_true.interpolated["data"][self.weight_key]

            return super().forward(pv, tv, pi, ti, pw, tw)

    return IndexMatchedData


class MatchedDistanceLoss(MSNELoss):
    def __init__(self, n_vertices: int | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_vertices = n_vertices

    def forward(self, y_pred: Surface, y_true: Surface):
        """Average (squared) distance between matched vertices of surfaces a and b.

        Parameters
        ----------
        surface_a : Surface
            First surface
        surface_b : Surface
            Second surface.

        Returns
        -------
        loss
            _description_
        """
        weights = (
            y_true.vertex_data["weights"] if "weights" in y_true.vertex_data else None
        )
        if self.n_vertices is None:
            x = y_pred.vertices
            y = y_true.vertices
        else:
            x = y_pred.vertices[:, self.n_vertices]
            y = y_true.vertices[:, self.n_vertices]
            if weights is not None:
                weights = weights[:, self.n_vertices]

        return super().forward(x, y, weights)


# class SurfaceRMSELoss(RMSELoss):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(self, y_pred: Surface, y_true: Surface):
#         """Average (squared) distance between matched vertices of surfaces a and b.

#         Parameters
#         ----------
#         surface_a : Surface
#             First surface
#         surface_b : Surface
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

SemiSymmetricNL1Loss = wrap_semi_symmetric(NL1Loss)
SampledSemiSymmetricNL1Loss = wrap_index_matched_data(SemiSymmetricNL1Loss)

SemiSymmetricNSELoss = wrap_semi_symmetric(NSELoss)
SampledSemiSymmetricNSELoss = wrap_index_matched_data(SemiSymmetricNSELoss)

SemiSymmetricCosSimLoss = wrap_semi_symmetric(MSCosSimLoss)
SampledSemiSymmetricCosSimLoss = wrap_index_matched_data(SemiSymmetricCosSimLoss)

SemiSymmetricNegLogLikLoss = wrap_semi_symmetric(NegLogLikLoss)
SampledSemiSymmetricNegLogLikLoss = wrap_index_matched_data(SemiSymmetricNegLogLikLoss)


# SemiHard
SemiSymmetricSemiHardNELoss = wrap_semi_symmetric(SemiHardSNELoss)
SampledSemiSymmetricSemiHardSNormLoss = wrap_index_matched_data(
    SemiSymmetricSemiHardNELoss
)


# REGULARIZATION


class FaceNormalConsistencyLoss(MSNELoss):  # MSCosSimLoss
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred):
        normals = y_pred.compute_face_normals()
        edge_face_normals = normals[:, y_pred.topology.face_adjacency]
        a, b = edge_face_normals.unbind(2)
        return super().forward(a, b)

        # integrate at each face

        # loss = torch.zeros((1, len(y_pred.faces)), device=y_pred.device)
        # error = sne(a,b)
        # loss = torch.index_add(
        #     loss,1,
        #     y_pred.topology.face_adjacency[:,0],
        #     error,
        # )
        # loss = torch.index_add(
        #     loss,1,
        #     y_pred.topology.face_adjacency[:,1],
        #     error,
        # )
        # loss = loss / 3.0


# import nibabel as nib
# import numpy as np
# from brainnet.mesh.surface import Surface

# v,f = nib.freesurfer.read_geometry("/mnt/scratch/personal/jesperdn/results/TopoFit/t1w_1mm/examples/epoch-00200.validation.surface.rh.white.true.00")
# v = torch.tensor(v.astype(np.float32))
# f = torch.tensor(f.astype(np.int32))
# surface = Surface(v,f)

# vv = surface.smooth_taubin(a=0.6,b=-0.61,n_iter=5)

# nib.freesurfer.write_geometry("test1", vv[0].numpy(), f.numpy())


class LaplacianLoss(MSNELoss):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.loss_fn = MSNELoss()

    def forward(self, surface: Surface):
        v = surface.vertices
        dv = torch.index_reduce(
            torch.zeros_like(v),
            self.dim,
            surface.topology.conv_index_reduce.long(),  # TODO needs to be long!?
            v.index_select(self.dim, surface.topology.conv_index_gather),
            "mean",
            include_self=False,
        )
        return self.loss_fn(dv, v)


class TaubinLoss(MSNELoss):
    def __init__(self, a=0.33, b=-0.34, n_iter: int = 10):
        super().__init__()
        self.taubin_kw = dict(a=a, b=b, n_iter=n_iter)
        self.loss_fn = MSNELoss()

    def forward(self, s: Surface):
        return self.loss_fn(s.vertices, s.smooth_taubin(**self.taubin_kw))


class IndexedSmoothnessLoss(MSNELoss):
    def __init__(self, dim=-1) -> None:
        """Penalize smoothness of the surface as measured by the Laplacian."""
        super().__init__(dim)

    def forward(self, y_pred, y_true):
        yi = unsqueeze_and_expand(y_true.vertex_data["chamfer_index"], y_pred.vertices)
        xi = unsqueeze_and_expand(y_pred.vertex_data["chamfer_index"], y_true.vertices)
        pred2true = y_pred.vertices.gather(1, yi)
        true2pred = y_true.vertices.gather(1, xi)

        pred2true_r = pred2true[:, y_true.topology.conv_index_reduce]
        pred2true_g = pred2true[:, y_true.topology.conv_index_gather]

        true2pred_r = true2pred[:, y_pred.topology.conv_index_reduce]
        true2pred_g = true2pred[:, y_pred.topology.conv_index_gather]

        return 0.5 * super().forward(pred2true_r, pred2true_g) + 0.5 * super().forward(
            true2pred_r, true2pred_g
        )

        # vi = y_pred.vertex_data["k2"].index_select(1, y_pred.topology.conv_index_reduce)
        # vj = y_pred.vertex_data["k2"].index_select(1, y_pred.topology.conv_index_gather)

        # ek3 = torch.index_reduce(
        #     torch.zeros((1, y_pred.topology.n_vertices), device=y_pred.device),
        #     1, y_pred.topology.conv_index_reduce,
        #     nl1(vi,vj),
        #     "mean", include_self=False,
        # )


class MeanCurvatureSmoothnessLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        """Penalize smoothness of the surface as measured by the Laplacian."""
        super().__init__(*args, **kwargs)

    def forward(self, y_pred):
        K = y_pred.compute_laplace_beltrami_operator()
        H = y_pred.compute_mean_curvature(K)
        ri, gi = y_pred.topology.get_convolution_indices()

        normalizer = torch.index_reduce(
            torch.zeros_like(H),
            1,
            ri,
            H_other.abs(),
            "mean",
            include_self=False,
        )
        H_self = H.index_select(1, ri)
        H_other = H.index_select(1, gi)
        error = torch.pow(H_self - H_other, 2.0)
        error = torch.pow(
            1.0 - H_other / torch.maximum(torch.ones_like(H_self), H_self.abs()), 2.0
        )
        return error.mean()

        x = torch.index_reduce(
            torch.zeros_like(H),
            1,
            ri,
            error,
            "amax",
            include_self=False,
        )


class VertexNormalLoss(MSCosSimLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred, y_true):
        n_pred = y_pred.compute_vertex_normals()
        n_true = y_true.compute_vertex_normals()


class SpringForceLoss(MSELoss):
    def __init__(self, *args, **kwargs) -> None:
        """Penalize differences in distances between neighboring vertices."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_distances(s):
        """Compute distances between a vertex and all its neighbors."""
        ri, gi = s.topology.get_convolution_indices()
        return torch.linalg.vector_norm(
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

    def forward(self, y_pred: dict[str, Surface]):
        vw = y_pred[self.inner].vertices
        vp = y_pred[self.outer].vertices
        nw = y_pred[self.inner].compute_vertex_normals()
        return super().forward(vp - vw, nw)

        # cos = super().forward(vp-vw, nw)

        # if self.cutoff is not None:
        #     cos_valid = cos[cos <= self.cutoff]
        #     # avoid empty array
        #     loss = (1 - cos_valid)**2 if cos_valid.any() else torch.tensor([0.0], device=cos_valid.device)
        # else:
        #     loss = (1 - cos)**2

        # return loss.mean() # cos, loss,


class EdgeNormLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, surface: Surface):
        """

        Returns
        -------
        loss : float
            Average variance over batch.
        """
        return surface.compute_edge_norm(unique=True).pow(2).mean()


class EdgeLengthVarianceLoss(MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, surfaces: Surface):
        """Variance of normalized edge length. Edge lengths are normalized so that
        their sum is equal to the number of edges. Otherwise, zero error can be
        achieved simply by shrinking the mesh.

        The idea is to encourage equilateral triangles.

        Parameters
        ----------
        surfaces : Surface


        Returns
        -------
        loss : float
            Average variance over batch.
        """
        edge_length = surfaces.compute_edge_norm(unique=True)
        n = edge_length.shape[1]
        # new mean is 1
        edge_length_mu1 = edge_length * n / edge_length.sum(1)[:, None]
        # MSE to the mean
        return super().forward(edge_length_mu1, 1)


# class TriangleLengthVarianceLoss(MSELoss):
#     def __init__(self):
#         super().__init__()

#     def forward(self, s: Surface):
#         """Variance of normalized edge length. Edge lengths are normalized so that
#         their sum is equal to the number of edges. Otherwise, zero error can be
#         achieved simply by shrinking the mesh.

#         The idea is to encourage equilateral triangles.

#         Parameters
#         ----------
#         surfaces : Surface


#         Returns
#         -------
#         loss : float
#             Average variance over batch.
#         """
#         edge_norm = self.compute_edge_norm()
#         # new mean *of each triangle* is 1
#         edge_norm_mu1 = 3.0 * edge_norm/edge_norm.sum(-1, keepdim=True)
#         # MSE to the mean
#         return super().forward(edge_norm_mu1, 1.0)


class TriangleQualityLoss(torch.nn.Module):
    def __init__(self):
        """The 'volume-length' quality metric for a triangle is defined as

            Q = 4*sqrt(3) * Area / sum(EdgeLengths**2)

        in table 6, row 4 of Shewchuk (2002).

        The loss is

            1.0 - mean(Q)

        References
        ----------
        Shewchuk (2002). What Is a Good Linear Finite Element? Interpolation,
            Conditioning, Anisotropy, and Quality Measures.
            https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf
        """
        super().__init__()

    def forward(self, s: Surface):
        a = 4 * torch.sqrt(torch.tensor(3.0, device=s.device))
        A = s.compute_face_areas()
        E = s.compute_edge_norm()
        # q=0 is worst
        # q=1 is best
        q = a * A / E.pow(2).sum(-1)
        return 1.0 - q.mean()  # 0 = best; 1 = worst


class AreaLoss(MSELoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, surface):
        area = surface.compute_face_areas()

        n = area.shape[1]
        # new mean is 1
        area_mu1 = area * n / area.sum(1)[:, None]

        return super().forward(area_mu1, 1)


# OTHER


class SelfIntersectionCount(torch.nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, surface: Surface):
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
