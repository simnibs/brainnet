from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses_surface import (
    EdgeLengthVarianceLoss,
    HingeLoss,
    MatchedAngleLoss,
    MatchedDistanceLoss,
    SymmetricChamferLoss,
    # SymmetricCurvatureNormLoss,
)

# Surface loss:
# - loss is averaged across lh and rh (as available)

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
        # curv=SurfaceSupervisedLoss(
        #     SymmetricCurvatureMSELoss(), y_pred="white", y_true="white",
        # ),
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
        # curv=SurfaceSupervisedLoss(
        #     SymmetricCurvatureMSELoss(), y_pred="pial", y_true="pial",
        # ),
    ),
    thickness=dict(
        angle=SurfaceRegularizationLoss(
            MatchedAngleLoss(inner="white", outer="pial", cosine_cutoff=0.5),
            y_pred = None, # pass everything through, i.e., both white and pial
        ),
    ),
)

head_weights = dict(white=1.0, pial=1.0, thickness=1.0)

loss_weights=dict(
        white=dict(matched = 1.0, hinge=100.0, edge=5.0, chamfer=0.0),
        pial=dict(matched = 1.0, hinge=100.0, edge=5.0, chamfer=0.0,),
        thickness=dict(angle = 1.0),
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
