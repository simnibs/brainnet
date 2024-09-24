import torch

from brainsynth.transforms import IntensityNormalization
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


# class MatchedDistanceLoss(torch.nn.MSELoss):
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


# class SymmetricMeanSquaredNormLoss(MeanSquaredNormLoss):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(
#         self,
#         y_pred: torch.Tensor,
#         y_true: torch.Tensor,
#         i_pred: torch.IntTensor,
#         i_true: torch.IntTensor,
#         w_pred: None | torch.Tensor = None,
#         w_true: None | torch.Tensor = None,
#     ):
#         """

#         i_pred contains indices into i_true and vice versa!

#         """

#         if (batch_size := y_pred.shape[0]) > 1:
#             batch_index = torch.arange(batch_size)[:, None]
#             return 0.5 * (
#                 super().forward(y_pred, y_true[batch_index, i_pred], w_pred)
#                 + super().forward(y_pred[batch_index, i_true], y_true, w_true)
#             )
#         else:
#             p = y_pred.squeeze(0)
#             t = y_true.squeeze(0)
#             w_pred = w_pred.squeeze(0) if w_pred is not None else w_pred
#             w_true = w_true.squeeze(0) if w_true is not None else w_true
#             return 0.5 * (
#                 super().forward(p, t[i_pred.squeeze(0)], w_pred)
#                 + super().forward(p[i_true.squeeze(0)], t, w_true)
#             )


# class SymmetricMSELoss(MSELoss):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(
#         self,
#         y_pred: torch.Tensor,
#         y_true: torch.Tensor,
#         i_pred: torch.IntTensor,
#         i_true: torch.IntTensor,
#         w_pred: None | torch.Tensor = None,
#         w_true: None | torch.Tensor = None,
#     ):
#         """

#         i_pred contains indices into i_true and vice versa!

#         """
#         if (batch_size := y_pred.shape[0]) > 1:
#             batch_index = torch.arange(batch_size)[:, None]
#             return 0.5 * (
#                 super().forward(y_pred, y_true[batch_index, i_pred], w_pred)
#                 + super().forward(y_pred[batch_index, i_true], y_true, w_true)
#             )
#         else:
#             p = y_pred.squeeze(0)
#             t = y_true.squeeze(0)
#             w_pred = w_pred.squeeze(0) if w_pred is not None else w_pred
#             w_true = w_true.squeeze(0) if w_true is not None else w_true
#             return 0.5 * (
#                 super().forward(p, t[i_pred.squeeze(0)], w_pred)
#                 + super().forward(p[i_true.squeeze(0)], t, w_true)
#             )


# class SymmetricNormalLoss(SymmetricMeanSquaredNormLoss):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(
#         self,
#         y_pred: TemplateSurfaces,
#         y_true: TemplateSurfaces,
#     ):
#         return super().forward(
#             y_pred.compute_vertex_normals(),
#             y_true.compute_vertex_normals(),
#             y_pred.vertex_data["index_sampled"],
#             y_true.vertex_data["index_sampled"],
#         )


# SymmetricChamferLoss and SymmetricCurvatureVectorLoss are the same except for
# the variable they grab in the surface class. This was done to keep the
# face consistent across losses...


class AsymmetricChamferLoss(MeanSquaredNormLoss):
    """Compute the mean squared distance between the sampled points in y_true
    and the corresponding (i.e., closest) sampled points in y_pred.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        ix = y_pred.batch_ix[:, None]
        return super().forward(
            y_pred.vertex_data["points_sampled"][
                ix, y_true.vertex_data["index_sampled"]
            ],
            y_true.vertex_data["points_sampled"],
        )


class SymmetricChamferLoss(SymmetricMeanSquaredNormLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        w_pred = (
            y_pred.vertex_data["weights_sampled"]
            if "weights_sampled" in y_pred.vertex_data
            else None
        )
        w_true = (
            y_true.vertex_data["weights_sampled"]
            if "weights_sampled" in y_true.vertex_data
            else None
        )
        return super().forward(
            y_pred.vertex_data["points_sampled"],
            y_true.vertex_data["points_sampled"],
            y_pred.vertex_data["index_sampled"],
            y_true.vertex_data["index_sampled"],
            w_pred,
            w_true,
            # broadcast against points_sampled
            # torch.clamp(1 + y_pred.vertex_data["H_sampled"].abs(), max=5)[..., None],
            # torch.clamp(1 + y_true.vertex_data["H_sampled"].abs(), max=5)[..., None],
        )


class AsymmetricCurvatureNormLoss(torch.nn.MSELoss):
    """Compute the mean squared distance between the sampled points in y_true
    and the corresponding (i.e., closest) sampled points in y_pred.

    Penalize the (signed) magnitude of the mean curvature.

    """

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        ix = y_pred.batch_ix[:, None]
        return super().forward(
            y_pred.vertex_data["H_sampled"][ix, y_true.vertex_data["index_sampled"]],
            y_true.vertex_data["H_sampled"],
        )


class SymmetricCurvatureNormLoss(SymmetricMeanSquaredNormLoss):
    def __init__(
        self, curv_key="H_sampled", index_key="index_sampled", *args, **kwargs
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



class SymmetricCurvatureMSELoss(SymmetricMSELoss):
    def __init__(
        self, curv_key="H_sampled", index_key="index_sampled", *args, **kwargs
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


class AsymmetricCurvatureAngleLoss(CosineSimilarityLoss):
    """Compute the mean squared distance between the sampled points in y_true
    and the corresponding (i.e., closest) sampled points in y_pred.

    Penalize the angles between the curvature vectors.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        ix = y_pred.batch_ix[:, None]
        return super().forward(
            y_pred.vertex_data["K_sampled"][ix, y_true.vertex_data["index_sampled"]],
            y_true.vertex_data["K_sampled"],
        )


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
            y_pred[self.inner].vertex_data["index_sampled"],
            y_true[self.inner].vertex_data["index_sampled"],
        ) + 0.5 * super().forward(
            yp_outer_th,
            yt_outer_th,
            y_pred[self.outer].vertex_data["index_sampled"],
            y_true[self.outer].vertex_data["index_sampled"],
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


# IMAGE LOSS FUNCTIONS


class GradientLoss(torch.nn.Module):
    def __init__(
        self,
        # spacing: float = 1.0, dim: list[int] | tuple = (2, 3, 4)
        n_spatial_dims: int = 3,
        device: str | torch.device = "cpu",
    ) -> None:
        """Gradient loss over entire image (the average gradient
        norm across the image).

        Parameters
        ----------
        spacing : int
            Voxel spacing.
        dim :
            Dimensions of input image over which to calculate the gradient.
        """
        super().__init__()
        device = torch.device(device) if isinstance(device, str) else device
        # self.spacing = spacing
        # self.dim = dim
        assert n_spatial_dims == 3
        self.n_spatial_dims = n_spatial_dims
        self.conv = getattr(torch.nn.functional, f"conv{n_spatial_dims}d")

        # filters to compute gradients along each dimension
        diff_op = torch.tensor([-1.0, 1.0], device=device)
        self.filters = tuple(
            diff_op.reshape(1, 1, *shape) for shape in ((2, 1, 1), (1, 2, 1), (1, 1, 2))
        )

    # def forward(self, image):
    #     # assume size is (N,C,W,H,D). When `image` is an SVF in 3D C=3.
    #     grad = torch.stack(torch.gradient(image, spacing=self.spacing, dim=self.dim))
    #     # stack to (3,N,C,W,H,D)
    #     return grad.norm(dim=0).mean()

    def forward(self, image):
        # compute grad along each dimension (x, y, z) for all channels
        n_channels = image.size()[1]
        loss = 0.0
        for f in self.filters:
            for c in range(n_channels):
                loss += self.conv(image[:, [c]], f).pow(2).mean()
        loss /= self.n_spatial_dims  # we one filter per spatial dim

        return loss


class MaskedGradientLoss(torch.nn.Module):
    def __init__(self, spacing: float = 1.0, method: str = "forward") -> None:
        """Gradient loss over selected voxels.

        Parameters
        ----------
        spacing : float, optional
            Voxel spacing.
        """
        super().__init__()
        self.spacing = spacing
        self.method = method

    def _clamp_slices_(self, indices: list[slice], minval=None, maxval=None):
        for s in indices:
            if minval is not None:
                s.start = min(s.start, minval)
            if maxval is not None:
                s.stop = max(s.stop, maxval)

    def _clamp_scatter_(self, indices: torch.Tensor, minval=None, maxval=None):
        for i in range(len(indices)):
            indices[i].clamp_(min=minval, max=maxval)

    def compute_gradient(self, image, indices):
        spatial_dims = image.size()[2:]
        match self.method:
            case "forward":
                # indices[0].clamp(max=spatial_dims[2]-1)
                # indices[1].clamp(max=spatial_dims[3]-1)
                # indices[2].clamp(max=spatial_dims[4]-1)
                if isinstance(indices, torch.Tensor):
                    self._clamp_scatter_(indices, maxval=spatial_dims - 1)
                elif isinstance(indices, list[slice]):
                    self._clamp_slices_(indices, maxval=spatial_dims - 1)

                f_x = image[:, :, *indices][None]
                f_xh = torch.stack(
                    (
                        image[:, :, indices[0] + 1, indices[1], indices[2]],
                        image[:, :, indices[0], indices[1] + 1, indices[2]],
                        image[:, :, indices[0], indices[1], indices[2] + 1],
                    )
                )

            case "backward":
                # indices[0].clamp(min=1)
                # indices[1].clamp(min=1)
                # indices[2].clamp(min=1)

                f_xh = image[:, :, *indices][None]
                f_x = torch.stack(
                    (
                        image[:, :, indices[0] - 1, indices[1], indices[2]],
                        image[:, :, indices[0], indices[1] - 1, indices[2]],
                        image[:, :, indices[0], indices[1], indices[2] - 1],
                    )
                )
            case "center":
                # indices[0].clamp(min=1, max=spatial_dims[2]-1)
                # indices[1].clamp(min=1, max=spatial_dims[3]-1)
                # indices[2].clamp(min=1, max=spatial_dims[4]-1)

                raise NotImplementedError
            case _:
                raise ValueError(f"Invalid method '{self.method}'")

        return (f_xh - f_x) / self.spacing  # (3, N, C, m)

    def _gradient_loss(self, grad):
        # return grad.abs().mean()
        return torch.mean(grad**2)

    def forward(self, image, indices):
        # N=2
        # C=3
        # dims = (3,4,5)
        # m = 4 # number of voxels to select
        # image = torch.rand((N,C,*dims))

        # indices = torch.stack([torch.randint(0,d,(m,)) for d in dims])

        # ignore edges
        # indices = torch.stack([torch.randint(1,d-1,(m,)) for d in dims])
        # indices = [slice(10,20), slice(2,15), slice(5,15)]
        # indices: (3, m)
        # selected_voxels = image[..., *indices]
        # selected_voxels = (N,C,m)

        # image: n,c,w,h,d
        # sel : n,c,m where m is the number of selected voxels
        grad = self.compute_gradient(image, indices)
        return self._gradient_loss(grad)


class NormalizedCrossCorrelationLoss(torch.nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        n_spatial_dims: int = 3,
        kernel_size: int | list[int] | tuple = 11,
        stride: int = 1,
        use_patch: bool = False,
        patch_size: None | list[int] | tuple = None,
        spatial_size: None | list[int] | tuple | torch.Size = None,
        ignore_ncc_sign: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        """

        Use a random patch because 7x7x7 convolutions are very slow on the
        full volume (e.g., 192x224x192)

                |-----------------------|
                ||-----------------|    |
                ||                 |    |
                ||        x        |    |
                ||                 |    |
                ||-----------------|    |
                |                       |
                |-----------------------|
        """
        super().__init__()
        self.ignore_ncc_sign = ignore_ncc_sign
        device = torch.device(device) if isinstance(device, str) else device
        if isinstance(kernel_size, int):
            kernel_size = n_spatial_dims * [kernel_size]
        assert all(k % 2 == 1 for k in kernel_size), "Only odd-sized kernels supported"
        self.kernel_offset = [int((k - 1) / 2) for k in kernel_size]
        self.kernel = torch.ones((n_channels, n_channels, *kernel_size), device=device)
        self.nw = self.kernel.sum()
        self.conv = getattr(torch.nn.functional, f"conv{n_spatial_dims}d")
        self.intensity_normalization = IntensityNormalization(high=0.5)  # 0.75
        self.stride = stride

        self.use_patch = use_patch
        if self.use_patch:
            assert patch_size is not None and spatial_size is not None
            assert all(
                i % 2 == 0 for i in patch_size
            )  # for now we only accept even sized sub-volumes
            self.patch_halfsize = tuple(int(i / 2) for i in patch_size)
            self.patch_limits = [
                (j, i - j) for i, j in zip(spatial_size, self.patch_halfsize)
            ]
        else:
            self.patch_halfsize = []
            self.patch_limits = []

    def sample_patch_slices(self):
        patch_center = tuple(torch.randint(i[0], i[1], (1,)) for i in self.patch_limits)
        return tuple(
            slice(c - h, c + h) for c, h in zip(patch_center, self.patch_halfsize)
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_valid: torch.Tensor | None = None,
        y_true_valid: torch.Tensor | None = None,
    ):
        """
        Returns
        -------
        _type_
            _description_
        """
        if self.use_patch:
            slicer = self.sample_patch_slices()
            x = y_pred[..., *slicer]
            y = y_true[..., *slicer]
        else:
            x = y_pred
            y = y_true

        # we don't need to form the fully expanded array, however, this can be
        # numerically unstable
        sum_x = self.conv(x, self.kernel, stride=self.stride)
        sum_y = self.conv(y, self.kernel, stride=self.stride)
        sum_x2 = self.conv(x**2, self.kernel, stride=self.stride)
        sum_y2 = self.conv(y**2, self.kernel, stride=self.stride)
        sum_xy = self.conv(x * y, self.kernel, stride=self.stride)

        if y_pred_valid is not None and y_true_valid is not None:
            if self.use_patch:
                valid = y_pred_valid[..., *slicer] & y_true_valid[..., *slicer]
            else:
                valid = y_pred_valid & y_true_valid

            # Compensate for kernel size and stride
            valid = valid[
                ...,
                self.kernel_offset[0] : -self.kernel_offset[0] : self.stride,
                self.kernel_offset[1] : -self.kernel_offset[1] : self.stride,
                self.kernel_offset[2] : -self.kernel_offset[2] : self.stride,
            ]

            sum_x = sum_x[valid]
            sum_y = sum_y[valid]
            sum_x2 = sum_x2[valid]
            sum_y2 = sum_y2[valid]
            sum_xy = sum_xy[valid]

        # This results in large numbers that need to be subtracted resulting in
        # overflows if using AMP (float16)
        # num = self.nw * sum_xy - sum_x * sum_y
        # denom_0 = self.nw * sum_x2 - sum_x**2
        # denom_1 = self.nw * sum_y2 - sum_y**2

        mean_x = sum_x / self.nw
        mean_y = sum_y / self.nw

        num = sum_xy + mean_x * mean_y * self.nw - mean_x * sum_y - mean_y * sum_x
        denom_0 = mean_x**2 * self.nw + sum_x2 - 2 * mean_x * sum_x
        denom_1 = mean_y**2 * self.nw + sum_y2 - 2 * mean_y * sum_y

        denom = denom_0 * denom_1
        valid = denom > 1e-5
        ncc = torch.zeros_like(denom)  # if num then we get a cast error when using AMP
        ncc[valid] = num[valid] / denom[valid].sqrt()

        invalid = (ncc < -1.0 - 1e-5) | (ncc > 1.0 + 1e-5)

        # q = valid.sum()/valid.numel() * 100.0
        # print(f"# VALID:        {q:10.2f}")
        # if q < 60.0:
        #     raise RuntimeError

        # print(f"# invalid: {invalid.sum()/invalid.numel():10.4f}")
        # print(ncc[invalid])

        # For now, just
        ncc[invalid] = 0

        # if invalid.any():
        #     indices = torch.stack(torch.where(invalid)).T

        #     select = ncc[invalid]
        #     select_clamp = select.clamp(-1.0, 1.0)

        #     invalid[invalid] = ~torch.isclose(select, select_clamp, atol=1e-3)

        # -1.0 <= ncc.mean() <= 1.0 or 0 <= ncc.abs.mean() <= 1.0 where 1.0 is
        # perfect
        ncc = ncc.abs() if self.ignore_ncc_sign else ncc

        all_valid = valid & ~invalid

        # weigh areas with actual image gradients more
        w = self.intensity_normalization(
            0.5 * (denom_0[all_valid] + denom_1[all_valid])
        )
        w = w / w.sum()
        return 1.0 - torch.sum(ncc[all_valid] * w)


def ScaledLoss(Loss):
    """Decorator to scaled input images before calculating the given loss."""

    class ScaledLoss(Loss):
        def __init__(
            self,
            interp_factor: float,
            interp_mode: str = "nearest",
            *args,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.interp_factor = float(interp_factor)
            self.interp_mode = interp_mode
            self.align_corners = (
                True
                if self.interp_mode in {"linear", "bilinear", "bicubic", "trilinear"}
                else None
            )

        def resample(self, image: torch.Tensor | None):
            if self.interp_factor == 1.0 or image is None:
                return image

            is_bool = image.dtype == torch.bool
            out = torch.nn.functional.interpolate(
                image.to(torch.uint8) if is_bool else image,
                scale_factor=self.interp_factor,
                mode=self.interp_mode,
                align_corners=self.align_corners,
            )
            return out.bool() if is_bool else out

        def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            y_pred_valid: torch.Tensor | None = None,
            y_true_valid: torch.Tensor | None = None,
        ):
            return super().forward(
                self.resample(y_pred),
                self.resample(y_true),
                self.resample(y_pred_valid),
                self.resample(y_true_valid),
            )

    return ScaledLoss


ScaledNormalizedCrossCorrelationLoss = ScaledLoss(NormalizedCrossCorrelationLoss)


# torch.corrcoef(torch.stack((f[win].ravel(), g[win].ravel())))


# torch.sum( (f[win]-f[win].mean()) * (g[win]-g[win].mean()) ) / (
#     f[win].std() * g[win].std()
# ) / self.nw

# x1 = f[win].sum()
# x2 = g[win].sum()
# x3 = (f[win]**2).sum()
# x4 = (g[win]**2).sum()
# x5 = (f[win] * g[win]).sum()

# idx = (1,0,116,66,177)
# idx1 = (1,0,idx[2]+2,idx[3]+2,idx[4]+2)

# win = (1,0,slice(idx1[2]-2,idx1[2]+3), slice(idx1[3]-2, idx1[3]+3), slice(idx1[4]-2, idx1[4]+3))
# win1 = (1,0,slice(idx[2]-2,idx[2]+3), slice(idx[3]-2, idx[3]+3), slice(idx[4]-2, idx[4]+3))

# mean_diff = []
# n = 10000
# for i in zip(torch.randint(10,180,(n,)),torch.randint(10,180,(n,)),torch.randint(10,180,(n,))):
#     idx = (1,0,i[0], i[1], i[2])
#     idx1 = (1,0,i[0]+2, i[1]+2, i[2]+2)

#     win = (1,0,slice(idx1[2]-2,idx1[2]+3), slice(idx1[3]-2, idx1[3]+3), slice(idx1[4]-2, idx1[4]+3))

#     q = torch.corrcoef(torch.stack((x[win].ravel(), y[win].ravel())))
#     # print(q)
#     # print(ncc[idx])
#     # print()
#     if torch.isnan(q).any():
#         continue
#     else:
#         mean_diff.append(torch.abs(ncc[idx] - q[0,1]))
#         if mean_diff[-1] > 0.1:
#             print(torch.cov(torch.stack((x[win].ravel(), y[win].ravel()))))

# mean_diff = torch.tensor(mean_diff)
# print(mean_diff.min())
# print(mean_diff.mean())
# print(mean_diff.max())

# idx = (1,0,106,181,30)
# idx1 = (1,0,106+2,181+2,30+2)

# win = (1,0,slice(106-2,106+3), slice(181-2, 181+3), slice(30-2, 30+3))
# win1 = (1,0,slice(106,106+3+2), slice(181, 181+3+2), slice(30, 30+3+2))


class DiceFromOneHotLoss(torch.nn.Module):
    def __init__(self, weigh_by_class_size: bool = True, ignore_background: bool = True) -> None:
        super().__init__()
        self.weigh_by_class_size = weigh_by_class_size
        self.ignore_background = ignore_background

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        # y_pred_valid: torch.Tensor | None = None,
        # y_true_valid: torch.Tensor | None = None,
    ):
        n, c = y_true.shape[:2]
        if self.ignore_background:
            y_pred = y_pred[:, 1:]
            y_true = y_true[:, 1:]
            c = c-1
        y_pred = y_pred.reshape(n, c, -1)
        y_true = y_true.reshape(n, c, -1)

        # if y_pred_valid is not None and y_true_valid is not None:
        #     valid = y_pred_valid.reshape(n, 1, -1) & y_true_valid.reshape(n, 1, -1)
        #     y_pred = y_pred[valid.expand(y_pred.shape)]
        #     y_true = y_true[valid.expand(y_true.shape)]

        intersection = torch.sum(y_pred & y_true, dim=-1)
        union = torch.sum(y_pred | y_true, dim=-1)
        nonzero = union > 0
        intersection = intersection[nonzero]
        union = union[nonzero]

        if self.weigh_by_class_size:
            class_size = y_true.sum(-1)
            # divide by batch size so that weights sum to 1
            weight = class_size / class_size.sum(-1, keepdim=True) / n
            weight = weight[nonzero]
            return 1.0 - torch.sum(weight * 2 * intersection / (union + intersection))
        else:
            return 1.0 - torch.mean(2 * intersection / (union + intersection))


class DiceFromLabelsLoss(torch.nn.Module):
    def __init__(self, num_classes: int, weigh_by_class_size: bool = True, ignore_background: bool = True) -> None:
        """Dice loss from label image (*not* one-hot encoded!).
        """
        super().__init__()
        self.num_classes = num_classes
        self.weigh_by_class_size = weigh_by_class_size
        self.ignore_background = ignore_background

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        # y_pred_valid: torch.Tensor | None = None,
        # y_true_valid: torch.Tensor | None = None,
    ):
        n, c = y_true.shape[:2]
        assert c == 1
        y_pred = y_pred.reshape(n, -1)
        y_true = y_true.reshape(n, -1)

        # Counts per class
        c_pred = torch.stack([i.bincount(minlength=self.num_classes) for i in y_pred])
        c_true = torch.stack([i.bincount(minlength=self.num_classes) for i in y_true])

        # It does not matter it we "bincount" y_pred or y_true as we only count
        # when there are equal
        intersection = torch.stack([i.bincount(weights=(i == j), minlength=self.num_classes) for i,j in zip(y_pred, y_true)])
        denom = c_pred + c_true
        if self.ignore_background:
            intersection = intersection[:, 1:]
            denom = denom[:, 1:]
            c_pred = c_pred[:, 1:]
            c_true = c_true[:, 1:]

        nonzero = denom > 0
        intersection = intersection[nonzero]
        denom = denom[nonzero]

        if self.weigh_by_class_size:
            # divide by batch size so that weights sum to 1
            weight = c_true / c_true.sum(-1, keepdim=True) / n
            weight = weight[nonzero]
            return 1.0 - torch.sum(weight * 2 * intersection / denom).float()
        else:
            return 1.0 - torch.mean(2 * intersection / denom).float()

# def compute_per_channel_dice(y_pred, y_true, epsilon=1e-6, weight=None):
#     """
#     Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given a multi channel input and target.
#     Assumes y_pred is a normalized probability, e.g. a result of Sigmoid or Softmax function.

#     Args:
#          y_pred (torch.Tensor): NxCxSpatial input tensor
#          y_true (torch.Tensor): NxCxSpatial target tensor
#          epsilon (float): prevents division by zero
#          weight (torch.Tensor): Cx1 tensor of weight per channel/class
#     """
#     assert y_pred.size() == y_true.size()

#     y_pred = flatten(y_pred)
#     y_true = flatten(y_true)
#     y_true = y_true.float()

#     # compute per channel Dice Coefficient
#     intersect = (y_pred * y_true).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect

#     # here we can use standard dice (y_pred + y_true).sum(-1) or extension (see V-Net) (y_pred^2 + y_true^2).sum(-1)
#     denominator = (y_pred * y_pred).sum(-1) + (y_true * y_true).sum(-1)
#     return 2 * (intersect / denominator.clamp(min=epsilon))


# class _AbstractDiceLoss(torch.nn.Module):
#     """
#     Base class for different implementations of Dice loss.
#     """

#     def __init__(self, weight=None, normalization='sigmoid'):
#         super(_AbstractDiceLoss, self).__init__()
#         self.register_buffer('weight', weight)
#         # The output from the network during training is assumed to be un-normalized probabilities and we would
#         # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
#         # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
#         # However if one would like to apply Softmax in order to get the proper probability distribution from the
#         # output, just specify `normalization=Softmax`
#         assert normalization in ['sigmoid', 'softmax', 'none']
#         if normalization == 'sigmoid':
#             self.normalization = torch.nn.Sigmoid()
#         elif normalization == 'softmax':
#             self.normalization = torch.nn.Softmax(dim=1)
#         else:
#             self.normalization = lambda x: x

#     def dice(self, y_pred, y_true, weight):
#         # actual Dice score computation; to be implemented by the subclass
#         raise NotImplementedError

#     def forward(self, y_pred, y_true):
#         y_pred = self.normalization(y_pred) # get probabilities from logits

#         per_channel_dice = self.dice(y_pred, y_true, weight=self.weight)

#         return 1. - per_channel_dice.mean()


# class DiceLoss(_AbstractDiceLoss):
#     """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
#     For multi-class segmentation `weight` parameter can be used to assign different weights per class.
#     The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
#     """

#     def __init__(self, weight=None, normalization='sigmoid'):
#         super().__init__(weight, normalization)

#     def dice(self, y_pred, y_true, weight):
#         return compute_per_channel_dice(y_pred, y_true, weight=self.weight)


# class GeneralizedDiceLoss(_AbstractDiceLoss):
#     """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
#     """

#     def __init__(self, normalization='sigmoid', epsilon=1e-6):
#         super().__init__(weight=None, normalization=normalization)
#         self.epsilon = epsilon

#     def dice(self, y_pred, y_true, weight):
#         assert y_pred.size() == y_true.size(), "'input' and 'target' must have the same shape"

#         y_pred = flatten(y_pred)
#         y_true = flatten(y_true)
#         y_true = y_true.float()

#         if y_pred.size(0) == 1:
#             # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
#             # put foreground and background voxels in separate channels
#             y_pred = torch.cat((y_pred, 1 - y_pred), dim=0)
#             y_true = torch.cat((y_true, 1 - y_true), dim=0)

#         # GDL weighting: the contribution of each label is corrected by the inverse of its volume
#         w_l = y_true.sum(-1)
#         w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
#         w_l.requires_grad = False

#         intersect = (y_pred * y_true).sum(-1)
#         intersect = intersect * w_l

#         denominator = (y_pred + y_true).sum(-1)
#         denominator = (denominator * w_l).clamp(min=self.epsilon)

#         return 2 * (intersect.sum() / denominator.sum())


# class BCEDiceLoss(torch.nn.Module):
#     """Linear combination of BCE and Dice losses"""

#     def __init__(self, alpha, beta):
#         super().__init__()
#         self.alpha = alpha
#         self.bce = torch.nn.BCEWithLogitsLoss()
#         self.beta = beta
#         self.dice = DiceLoss()

#     def forward(self, y_pred, y_true):
#         return self.alpha * self.bce(y_pred, y_true) + self.beta * self.dice(y_pred, y_true)


# class WeightedCrossEntropyLoss(torch.nn.Module):
#     """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
#     """

#     def __init__(self, ignore_index=-1):
#         super().__init__()
#         self.ignore_index = ignore_index

#     def forward(self, y_pred, y_true):
#         weight = self._class_weights(y_pred)
#         return torch.nn.functional.cross_entropy(y_pred, y_true, weight=weight, ignore_index=self.ignore_index)

#     @staticmethod
#     def _class_weights(y_pred):
#         # normalize the input first
#         y_pred = torch.nn.functional.softmax(y_pred, dim=1)
#         flattened = flatten(y_pred)
#         nominator = (1. - flattened).sum(-1)
#         denominator = flattened.sum(-1)
#         class_weights = Variable(nominator / denominator, requires_grad=False)
#         return class_weights


# class PixelWiseCrossEntropyLoss(torch.nn.Module):
#     def __init__(self, class_weights=None, ignore_index=None):
#         super().__init__()
#         self.register_buffer('class_weights', class_weights)
#         self.ignore_index = ignore_index
#         self.log_softmax = torch.nn.LogSoftmax(dim=1)

#     def forward(self, y_pred, y_true, weights):
#         assert y_true.size() == weights.size()
#         # normalize the input
#         log_probabilities = self.log_softmax(y_pred)
#         # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
#         y_true = expand_as_one_hot(y_true, C=y_pred.size()[1], ignore_index=self.ignore_index)
#         # expand weights
#         weights = weights.unsqueeze(1)
#         weights = weights.expand_as(y_pred)

#         # use default class_weights if none is given
#         class_weights = torch.ones(y_pred.size()[1]).float().cuda() if self.class_weights is None else self.class_weights

#         # resize class_weights to be broadcastable into the weights
#         class_weights = class_weights.view(1, -1, 1, 1, 1)

#         weights = class_weights * weights

#         result = -weights * y_true * log_probabilities

#         return result.mean()


# class WeightedSmoothL1Loss(torch.nn.SmoothL1Loss):
#     def __init__(self, threshold, initial_weight, apply_below_threshold=True):
#         super().__init__(reduction="none")
#         self.threshold = threshold
#         self.apply_below_threshold = apply_below_threshold
#         self.weight = initial_weight

#     def forward(self, y_pred, y_true):
#         l1 = super().forward(y_pred, y_true)

#         mask = y_true < self.threshold if self.apply_below_threshold else y_true >= self.threshold
#         l1[mask] = l1[mask] * self.weight

#         return l1.mean()

# def flatten(tensor):
#     """Flattens a given tensor such that the channel axis is first.
#     The shapes are transformed as follows:
#        (N, C, D, H, W) -> (C, N * D * H * W)
#     """
#     # number of channels
#     C = tensor.size(1)
#     # new axis order
#     axis_order = (1, 0) + tuple(range(2, tensor.dim()))
#     # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
#     transposed = tensor.permute(axis_order)
#     # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
#     return transposed.contiguous().view(C, -1)
