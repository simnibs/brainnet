from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    SupervisedLoss,
)
from brainnet.modules.losses_surface import (
    EdgeLengthVarianceLoss,
    FaceNormalConsistencyLoss,
    MatchedDistanceLoss,
    TriangleLengthVarianceLoss,
)


functions = dict(
    white=dict(
        matched=SupervisedLoss(MatchedDistanceLoss(), y_pred="white", y_true="white"),
        spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="white"),
        edge_var=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        tri_var=SurfaceRegularizationLoss(TriangleLengthVarianceLoss(), y_pred="white"),
    ),
)

head_weights = dict(white=1.0)

loss_weights = dict(
    white=dict(
        edge_var=20.0,
        matched=1.0,
        spring=10.0,
        tri_var=40.0,
    ),
)

cfg_loss = LossParameters(functions, head_weights, loss_weights)
