import torch

from brainnet.mesh.surface import TemplateSurfaces


# SURFACE LOSS FUNCTIONS


class MSELoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b, w=None):
        if w is None:
            return torch.mean((a - b) ** 2)
        else:
            return torch.sum(w * (a - b) ** 2) / w.sum()


# Norm loss = 3 * MSE loss


class MeanSquaredNormLoss(torch.nn.Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim
        # reduction="mean"
        # self.reduction = reduction

    def forward(self, a, b, w=None):
        # getattr(T, reduction)()
        if w is None:
            return torch.sum((a - b) ** 2, self.dim).mean()
        else:
            return torch.sum(w * torch.sum((a - b) ** 2, self.dim)) / w.sum()


class CosineSimilarityLoss(torch.nn.CosineSimilarity):
    def __init__(self, dim: int = -1, eps: float = 1e-8) -> None:
        super().__init__(dim, eps)

    def forward(self, a, b):
        return torch.mean((1 - super().forward(a, b)) ** 2)


class MatchedDistanceLoss(MeanSquaredNormLoss):

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


class MatchedCurvatureMSELoss(MSELoss):
    def __init__(self, curv_key="H", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.curv_key = curv_key

    def forward(self, y_pred: TemplateSurfaces, y_true: TemplateSurfaces):
        return super().forward(
            y_pred.vertex_data[self.curv_key], y_true.vertex_data[self.curv_key]
        )


class MatchedCurvatureNormLoss(MeanSquaredNormLoss):
    def __init__(self, curv_key="H", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred: TemplateSurfaces, y_true: TemplateSurfaces):
        return super().forward(y_pred.vertex_data["H"], y_true.vertex_data["H"])


def SymmetricLoss(Loss):
    """Decorator to scaled input images before calculating the given loss."""

    class SymmetricLoss(Loss):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

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

            i_pred contains indices into i_true and vice versa!

            """

            if (batch_size := y_pred.shape[0]) > 1:
                batch_index = torch.arange(batch_size)[:, None]
                return 0.5 * (
                    super().forward(y_pred, y_true[batch_index, i_pred], w_pred)
                    + super().forward(y_pred[batch_index, i_true], y_true, w_true)
                )
            else:
                p = y_pred.squeeze(0)
                t = y_true.squeeze(0)
                w_pred = w_pred.squeeze(0) if w_pred is not None else w_pred
                w_true = w_true.squeeze(0) if w_true is not None else w_true
                return 0.5 * (
                    super().forward(p, t[i_pred.squeeze(0)], w_pred)
                    + super().forward(p[i_true.squeeze(0)], t, w_true)
                )

    return SymmetricLoss

SymmetricMeanSquaredNormLoss = SymmetricLoss(MeanSquaredNormLoss)
SymmetricMSELoss = SymmetricLoss(MSELoss)

class SymmetricSampledNormLoss(SymmetricMeanSquaredNormLoss):
    def __init__(
        self, value_key: str, index_key="sampled_index", weight_key: str | None = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.value_key = value_key
        self.index_key = index_key
        self.weight_key = weight_key

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        if self.weight_key is None:
            return super().forward(
                y_pred.vertex_data[self.value_key],
                y_true.vertex_data[self.value_key],
                y_pred.vertex_data[self.index_key],
                y_true.vertex_data[self.index_key],
            )

        else:
            return super().forward(
                y_pred.vertex_data[self.value_key],
                y_true.vertex_data[self.value_key],
                y_pred.vertex_data[self.index_key],
                y_true.vertex_data[self.index_key],
                y_pred.vertex_data[self.weight_key],
                y_true.vertex_data[self.weight_key],
            )


class SymmetricSampledMSELoss(SymmetricMSELoss):
    def __init__(
        self, value_key: str, index_key="sampled_index", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.value_key = value_key
        self.index_key = index_key

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        return super().forward(
            y_pred.vertex_data[self.value_key],
            y_true.vertex_data[self.value_key],
            y_pred.vertex_data[self.index_key],
            y_true.vertex_data[self.index_key],
        )


class SymmetricNormalMSELoss(SymmetricMSELoss):
    def __init__(
        self, curv_key="normals_sampled", index_key="sampled_index", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.curv_key = curv_key
        self.index_key = index_key

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        return super().forward(
            y_pred.vertex_data[self.curv_key],
            y_true.vertex_data[self.curv_key],
            y_pred.vertex_data[self.index_key],
            y_true.vertex_data[self.index_key],
        )


# class AsymmetricCurvatureAngleLoss(CosineSimilarityLoss):
#     """Compute the mean squared distance between the sampled points in y_true
#     and the corresponding (i.e., closest) sampled points in y_pred.

#     Penalize the angles between the curvature vectors.

#     """

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(
#         self,
#         y_pred: TemplateSurfaces,
#         y_true: TemplateSurfaces,
#     ):
#         ix = y_pred.batch_ix[:, None]
#         return super().forward(
#             y_pred.vertex_data["K_sampled"][ix, y_true.vertex_data["sampled_index"]],
#             y_true.vertex_data["K_sampled"],
#         )


class HingeLoss(MeanSquaredNormLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred):
        normals = y_pred.compute_face_normals()
        edge_face_normals = normals[:, y_pred.topology.face_adjacency]
        a, b = edge_face_normals.unbind(2)
        return super().forward(a, b)


# class VertexToVertexAngleLoss(torch.nn.CosineSimilarity):

#     def __init__(self, cosine_cutoff = None, dim: int = 2) -> None:
#         super().__init__(dim=dim)

#         self.inner = "white"
#         self.outer = "pial"
#         self.cosine_cutoff = cosine_cutoff

#     def forward(self, y_pred: dict[str, TemplateSurfaces]):
#         vw = y_pred[self.inner].vertices
#         vp = y_pred[self.outer].vertices

#         nw = y_pred[self.inner].compute_vertex_normals()

#         cos = super().forward(vp-vw, nw)

#         if self.cosine_cutoff is not None:
#             cos_valid = cos[cos <= self.cosine_cutoff]
#             # avoid empty array
#             loss = (1 - cos_valid)**2 if cos_valid.any() else torch.tensor([0.0], device=cos_valid.device)
#         else:
#             loss = (1 - cos)**2

#         return cos, loss, loss.mean()


class MatchedAngleLoss(torch.nn.CosineSimilarity):

    def __init__(
        self,
        inner: str = "white",
        outer: str = "pial",
        cosine_cutoff: None | float = None,
        dim: int = 2,
    ) -> None:
        super().__init__(dim=dim)

        self.inner = inner
        self.outer = outer
        self.cosine_cutoff = cosine_cutoff

    def forward(self, y_pred: dict[str, TemplateSurfaces]):
        vw = y_pred[self.inner].vertices
        vp = y_pred[self.outer].vertices

        nw = y_pred[self.inner].compute_vertex_normals()

        cos = super().forward(vp - vw, nw)

        if self.cosine_cutoff is not None:
            cos_valid = cos[cos <= self.cosine_cutoff]
            loss = (1 - cos_valid) ** 2
            # avoid empty array
            if loss.numel() == 0:
                loss = torch.tensor([0.0], device=cos_valid.device)
        else:
            loss = (1 - cos) ** 2

        return loss.mean()


class SymmetricThicknessLoss(SymmetricMeanSquaredNormLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.inner = "white"
        self.outer = "pial"

    def compute_thickness_vector(self, y):

        y_in = y[self.inner]
        y_out = y[self.outer]
        ref = y_in  # arbitrary

        # these are indices into y_pred[h][outer]!
        index = ref.nearest_neighbor_tensors(
            y_in.vertex_data["points_sampled"],
            y_out.vertex_data["points_sampled"],
        )
        ix = y_out.batch_ix[:, None]
        y_inner_th_vec = (
            y_out.vertex_data["points_sampled"][ix, index]
            - y_in.vertex_data["points_sampled"]
        )

        # these are indices into y_pred[h][inner]!
        index = ref.nearest_neighbor_tensors(
            y_out.vertex_data["points_sampled"],
            y_in.vertex_data["points_sampled"],
        )
        ix = y_in.batch_ix[:, None]
        y_outer_th_vec = (
            y_out.vertex_data["points_sampled"]
            - y_in.vertex_data["points_sampled"][ix, index]
        )

        # inner_th_vec
        #   vector from sampled points on INNER surface to closest
        #   sampled point on OUTER surface
        # outer_th_vec
        #   vector from sampled points on OUTER surface to closest
        #   sampled point on INNER surface

        return y_inner_th_vec, y_outer_th_vec

    def forward(self, y_pred, y_true):
        """Compute the thickness vectors"""

        yp_inner_th, yp_outer_th = self.compute_thickness_vector(y_pred)
        yt_inner_th, yt_outer_th = self.compute_thickness_vector(y_true)

        return 0.5 * super().forward(
            yp_inner_th,
            yt_inner_th,
            y_pred[self.inner].vertex_data["sampled_index"],
            y_true[self.inner].vertex_data["sampled_index"],
        ) + 0.5 * super().forward(
            yp_outer_th,
            yt_outer_th,
            y_pred[self.outer].vertex_data["sampled_index"],
            y_true[self.outer].vertex_data["sampled_index"],
        )


# class CurvatureWeightedHingeLoss(MeanSquaredNormLoss):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(self, y_pred, y_true):
#         y_true.vertex_data["face_absH"]
#         y_true.vertex_data["index"]
#         y_pred.topology.faces_to_edges

#         normals = y_pred.compute_face_normals()
#         edge_face_normals = normals[:, y_pred.topology.face_adjacency]
#         a, b = edge_face_normals.unbind(2)
#         return super().forward(a, b)


class EdgeLengthVarianceLoss(torch.nn.Module):
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
        # mean squared difference to the mean
        return torch.mean(torch.sum((edge_length_mu1 - 1) ** 2, 1) / n)


class SelfIntersectionCount(torch.nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, surface: TemplateSurfaces):
        if surface.n_batch == 1:
            _, n = surface.compute_self_intersections()
        else:
            n = torch.mean(torch.stack([i for _,i in surface.compute_self_intersections()]).float())
        return n / surface.topology.n_vertices if self.normalize else n
