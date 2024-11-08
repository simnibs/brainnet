from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses_surface import (
    EdgeLengthVarianceLoss,
    HingeLoss,
    MatchedDistanceLoss,
    MatchedCurvatureNormLoss,
    SelfIntersectionCount,
)

# Surface loss:
# - loss is averaged across lh and rh (as available)


kw_inflated = dict(y_pred="inflated", y_true="inflated")

functions = dict(
    inflated=dict(
        distance=SurfaceSupervisedLoss(MatchedDistanceLoss(), **kw_inflated),
        curv=SurfaceSupervisedLoss(MatchedCurvatureNormLoss(), **kw_inflated),
        # hinge=SurfaceRegularizationLoss(HingeLoss(), y_pred="white"),
        # edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="inflated"),
    ),
)

head_weights = dict(inflated=1.0)

loss_weights = dict(
    inflated=dict(distance=1.0, curv=1.0, sif=0.0),
)

cfg_loss = LossParameters(functions, head_weights, loss_weights)
