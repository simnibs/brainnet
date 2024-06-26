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
        weights = y_true.vertex_data["weights"] if "weights" in y_true.vertex_data else None
        return super().forward(y_pred.vertices, y_true.vertices, weights)


class MatchedCurvatureLoss(MeanSquaredNormLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred: TemplateSurfaces, y_true: TemplateSurfaces):
        return super().forward(y_pred.vertex_data["H"], y_true.vertex_data["H"])


# SymmetricChamferLoss and SymmetricCurvatureVectorLoss are the same except for
# the variable they grab in the surface class. This was done to keep the
# face consistent across losses...


class SymmetricMeanSquaredNormLoss(MeanSquaredNormLoss):
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

class SymmetricMSELoss(MSELoss):
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


class SymmetricChamferLoss(SymmetricMeanSquaredNormLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        w_pred = y_pred.vertex_data["weights_sampled"] if "weights_sampled" in y_pred.vertex_data else None
        w_true = y_true.vertex_data["weights_sampled"] if "weights_sampled" in y_true.vertex_data else None
        return super().forward(
            y_pred.vertex_data["points_sampled"],
            y_true.vertex_data["points_sampled"],
            y_pred.vertex_data["index_sampled"],
            y_true.vertex_data["index_sampled"],
            w_pred,
            w_true,
            # broadcast against points_sampled
            #torch.clamp(1 + y_pred.vertex_data["H_sampled"].abs(), max=5)[..., None],
            #torch.clamp(1 + y_true.vertex_data["H_sampled"].abs(), max=5)[..., None],
        )


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


# class AsymmetricCurvatureLoss(MeanSquaredNormLoss):
#     """Compute the mean squared distance between the sampled points in y_true
#     and the corresponding (i.e., closest) sampled points in y_pred.
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
#             y_pred.vertex_data["K_sampled"][ix, y_true.vertex_data["index_sampled"]],
#             y_true.vertex_data["K_sampled"],
#         )


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


class SymmetricCurvatureNormLoss(SymmetricMSELoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        y_pred: TemplateSurfaces,
        y_true: TemplateSurfaces,
    ):
        return super().forward(
            y_pred.vertex_data["H_sampled"],
            y_true.vertex_data["H_sampled"],
            y_pred.vertex_data["index_sampled"],
            y_true.vertex_data["index_sampled"],
        )


class MatchedCurvatureNormLoss(MeanSquaredNormLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred: TemplateSurfaces, y_true: TemplateSurfaces):
        return super().forward(y_pred.vertex_data["H"], y_true.vertex_data["H"])


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


# class SymmetricCurvatureLoss(SymmetricMeanSquaredNormLoss):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(
#         self,
#         y_pred: TemplateSurfaces,
#         y_true: TemplateSurfaces,
#     ):
#         return super().forward(
#             y_pred.vertex_data["curv"],
#             y_true.vertex_data["curv"],
#             y_pred.vertex_data["index"],
#             y_true.vertex_data["index"],
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

    def __init__(self, inner: str = "white", outer: str = "pial", cosine_cutoff: None | float = None, dim: int = 2) -> None:
        super().__init__(dim=dim)

        self.inner = inner
        self.outer = outer
        self.cosine_cutoff = cosine_cutoff

    def forward(self, y_pred: dict[str, TemplateSurfaces]):
        vw = y_pred[self.inner].vertices
        vp = y_pred[self.outer].vertices

        nw = y_pred[self.inner].compute_vertex_normals()

        cos = super().forward(vp-vw, nw)

        if self.cosine_cutoff is not None:
            cos_valid = cos[cos <= self.cosine_cutoff]
            loss = (1 - cos_valid)**2
            # avoid empty array
            if loss.numel() == 0:
                loss = torch.tensor([0.0], device=cos_valid.device)
        else:
            loss = (1 - cos)**2

        return loss.mean()

class SymmetricThicknessLoss(SymmetricMeanSquaredNormLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.inner = "white"
        self.outer = "pial"


    def compute_thickness_vector(self, y):

        y_in = y[self.inner]
        y_out = y[self.outer]
        ref = y_in # arbitrary

        # these are indices into y_pred[h][outer]!
        index = ref.nearest_neighbor_tensors(
            y_in.vertex_data["points_sampled"],
            y_out.vertex_data["points_sampled"],
        )
        ix = y_out.batch_ix[:, None]
        y_inner_th_vec = y_out.vertex_data["points_sampled"][ix, index] - y_in.vertex_data["points_sampled"]

        # these are indices into y_pred[h][inner]!
        index = ref.nearest_neighbor_tensors(
            y_out.vertex_data["points_sampled"],
            y_in.vertex_data["points_sampled"],
        )
        ix = y_in.batch_ix[:, None]
        y_outer_th_vec = y_out.vertex_data["points_sampled"] - y_in.vertex_data["points_sampled"][ix, index]

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
