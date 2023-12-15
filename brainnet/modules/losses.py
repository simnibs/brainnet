import torch
from torch.autograd import Variable

from brainnet.mesh.surface import BatchedSurfaces


# # Image (Tensor) input
# IMAGE_LOSS = {}

# # Surface input
# SURFACE_LOSS = {
#     EdgeLengthVarianceLoss,
#     MatchedDistanceLoss,
#     SymmetricChamferLoss,
#     SymmetricCurvatureVectorLoss,
# }
# # Only operates on the predicted values
# REGULARIZATION_LOSS = {EdgeLengthVarianceLoss, GradientSmoothness}


# SURFACE LOSS FUNCTIONS

class MatchedDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, surface_a: BatchedSurfaces, surface_b: BatchedSurfaces):
        """Average (squared) distance between matched vertices of surfaces a and b.

        Parameters
        ----------
        surface_a : BatchedSurfaces
            First surface
        surface_b : BatchedSurfaces
            Second surface.

        Returns
        -------
        loss
            _description_
        """
        return self.mse_loss(surface_a.vertices, surface_b.vertices)

# SymmetricChamferLoss and SymmetricCurvatureVectorLoss are the same except for
# the variable they grab in the surface class. This was done to keep the
# face consistent across losses...

class SymmetricMSELoss(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            i_pred: torch.IntTensor,
            i_true: torch.IntTensor
        ):
        return 0.5 * (super().forward(y_pred, y_true[i_true]) + super().forward(y_pred[i_pred], y_true))


class SymmetricChamferLoss(SymmetricMSELoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
            self,
            y_pred: BatchedSurfaces,
            y_true: BatchedSurfaces,
            i_pred: torch.IntTensor | None = None,
            i_true: torch.IntTensor | None = None,
        ):
        # nearest neighbor
        i_pred = y_true.nearest_neighbor(y_pred) if i_pred is None else i_pred
        i_true = y_pred.nearest_neighbor(y_true) if i_true is None else i_true

        return super().forward(y_pred.vertices, y_true.vertices, i_pred, i_true)

class SymmetricCurvatureLoss(SymmetricMSELoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_smooth = 3

    def forward(
            self,
            y_pred: BatchedSurfaces,
            y_true: BatchedSurfaces,
            i_pred: torch.IntTensor | None = None,
            i_true: torch.IntTensor | None = None,
            curv_true: torch.Tensor | None = None,
        ):
        # nearest neighbor
        i_pred = y_true.nearest_neighbor(y_pred) if i_pred is None else i_pred
        i_true = y_pred.nearest_neighbor(y_true) if i_true is None else i_true

        # curvature
        curv_pred = y_pred.compute_mean_curvature_vector()
        if curv_true is None:
            curv_true = y_true.compute_mean_curvature_vector()
            curv_true = y_true.compute_iterative_spatial_smoothing(curv_true, self.n_smooth)

        return super().forward(curv_pred, curv_true, i_pred, i_true)



# class SymmetricChamferLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse_loss = torch.nn.MSELoss()

#     def forward(self, surface_a: BatchedSurfaces, surface_b: BatchedSurfaces) -> float:
#         """Symmetric chamfer loss between surfaces a and b, i.e., the average of
#         the (squared) distance between closest points in surfaces a and b (from a
#         to b and vice versa).

#         Parameters
#         ----------
#         surface_a : _type_
#             _: B
#         surface_b : _type_
#             _description_

#         Returns
#         -------
#         loss : float
#             Symmetric chamfer loss.
#         """
#         vertices_b = surface_b.vertices[surface_b.vertex_data["nn_index"]]
#         vertices_a = surface_a.vertices[surface_a.vertex_data["nn_index"]]

#         loss_a = self.mse_loss(surface_a.vertices, vertices_b)
#         loss_b = self.mse_loss(surface_b.vertices, vertices_a)

#         return 0.5 * (loss_a + loss_b)


# class SymmetricCurvatureLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse_loss = torch.nn.MSELoss()

#     def forward(self, surface_a: BatchedSurfaces, surface_b: BatchedSurfaces):
#         """Mean squared error of the mean curvature vector of the predicted and the
#         target surface. The target surface curvature vectors should be slightly
#         smoothed to minimize noise as they are effectively the prior of the
#         curvature.

#         Assumes that the curvature information is placed in a vertex_data field
#         called `curv`.

#         Assumes that the nearest neighbor information (i.e., a ) about the
#         """
#         # assert "curv" in surface_a.vertex_data, "Curvature information should be stored in .vertex_data['curv']"
#         # assert "curv" in surface_b.vertex_data, "Curvature information should be stored in .vertex_data['curv']"
#         curv_b = surface_b.vertex_data["curv"][surface_b.vertex_data["nn_index"]]
#         curv_a = surface_a.vertex_data["curv"][surface_a.vertex_data["nn_index"]]

#         loss_a = self.mse_loss(surface_a.vertex_data["curv"], curv_b)
#         loss_b = self.mse_loss(surface_b.vertex_data["curv"], curv_a)

#         return 0.5 * (loss_a + loss_b)


class EdgeLengthVarianceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, surfaces: BatchedSurfaces):
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
        edge_length = surfaces.compute_edge_lengths()
        n = edge_length.shape[1]
        edge_length *= n/edge_length.sum() # new mean is 1
        return torch.sum((edge_length-1)**2)/n # variance


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
