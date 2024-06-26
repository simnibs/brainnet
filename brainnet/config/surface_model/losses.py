# Segmentation

# seg:
#   DiceCE:
#     module:
#       name: SupervisedLoss
#       kwargs: # initialization kwargs for module
#         y_pred: segmentation
#         y_true: segmentation
#     loss:
#       name: monai.losses.DiceCELoss
#       kwargs: # initialization kwargs for loss function
#         # include_background: false  # default = true
#         softmax: true     # apply softmax before dice loss
#         # lambda_dice: 0.5  # default = 1.0
#         # lambda_ce: 0.5    # default = 1.0

# seg:
#   DiceCE:
#     module:
#       name: MaskedSupervisedLoss
#       kwargs: # initialization kwargs for module
#         y_pred: segmentation
#         y_true: segmentation
#     loss:
#       name: monai.losses.DiceCELoss
#       kwargs: # initialization kwargs for loss function
#         # include_background: false  # default = true
#         softmax: true     # apply softmax before dice loss
#         # lambda_dice: 0.5  # default = 1.0
#         # lambda_ce: 0.5    # default = 1.0


# Surface loss:
# - loss is averaged across lh and rh (as available)

from brainnet.config.base import LossParameters

from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses import (
    EdgeLengthVarianceLoss,
    HingeLoss,
    MatchedAngleLoss,
    MatchedDistanceLoss,
    SymmetricChamferLoss,
    SymmetricCurvatureNormLoss,
)

functions = dict(
    white=dict(
        matched=SurfaceSupervisedLoss(
            MatchedDistanceLoss(), y_pred="white", y_true="white",
        ),
        hinge=SurfaceRegularizationLoss(HingeLoss(), y_pred="white"),
        edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        chamfer=SurfaceSupervisedLoss(
            SymmetricChamferLoss(), y_pred="white", y_true="white",
        ),
        curv=SurfaceSupervisedLoss(
            SymmetricCurvatureNormLoss(), y_pred="white", y_true="white",
        ),
    ),
    pial=dict(
        matched=SurfaceSupervisedLoss(
            MatchedDistanceLoss(), y_pred="pial", y_true="pial",
        ),
        hinge=SurfaceRegularizationLoss(HingeLoss(), y_pred="pial"),
        edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="pial"),
        chamfer=SurfaceSupervisedLoss(
            SymmetricChamferLoss(), y_pred="pial", y_true="pial",
        ),
        curv=SurfaceSupervisedLoss(
            SymmetricCurvatureNormLoss(), y_pred="pial", y_true="pial",
        ),
    ),
    thickness=dict(
        angle=SurfaceRegularizationLoss(
            MatchedAngleLoss(inner="white", outer="pial", cosine_cutoff=0.5),
        ),
    ),
)

head_weights = dict(seg=1.0, white=1.0, pial=1.0, cross=1.0)

loss_weights=dict(
        seg=dict(DiceCE=1.0),
        white=dict(matched = 1.0, hinge=100.0, edge=5.0, chamfer=0.0, curv=0.0),
        pial=dict(matched = 1.0, hinge=100.0, edge=5.0, chamfer=0.0, curv=0.0)
    )

#           1     200     400     600     800     1000    1200    1400    ... 1800
# pred res  4     4       4       4       4       5       5       5       5
# ----------------------------------------------------------------------------------
# matched   1.0   1.0     1.0     0.01    0.01    0.001            0.0        0.0
# hinge   100.0  10.0    10.0     0.1     0.1     0.01             0.0        0.0
# edge     10.0  10.0    10.0    10.0    10.0     10.0             5.0        2.5
# chamfer                         1.0     1.0     1.0      1.0     1.0        1.0
# curv                           50.0    50.0    10.0     10.0    10.0        5.0/2.5
#

cfg_loss = LossParameters(functions, head_weights, loss_weights)
