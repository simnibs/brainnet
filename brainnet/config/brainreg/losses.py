from brainnet.config.base import LossParameters

from brainnet.modules.loss_wrappers import (
    # SurfaceRegularizationLoss,
    RegularizationLoss,
    SupervisedLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses import (
    # EdgeLengthVarianceLoss,
    # MatchedDistanceLoss,
    GradientLoss,
    NormalizedCrossCorrelationLoss,
    SymmetricChamferLoss,
    SymmetricCurvatureNormLoss,
)

functions = dict(
    svf=dict(gradient=RegularizationLoss(GradientLoss(), y_pred="svf")),
    image=dict(
        ncc=SupervisedLoss(
            # masked!
            # we need to set the device for this loss
            NormalizedCrossCorrelationLoss(device="cuda"),
            y_pred="t1w_areg_mni",
            y_true="t1w_areg_mni",
            # y_pred_mask="t1w_areg_mni_mask"
            # y_true_mask="t1w_areg_mni_mask"
        )
    ),
    white=dict(
        # edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="surface"),
        chamfer=SurfaceSupervisedLoss(
            SymmetricChamferLoss(), y_pred="white", y_true="white"
        ),
        curv=SurfaceSupervisedLoss(
            SymmetricCurvatureNormLoss(),
            y_pred="white",
            y_true="white",
        ),
    ),
    pial=dict(
        # edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="surface"),
        chamfer=SurfaceSupervisedLoss(
            SymmetricChamferLoss(),
            y_pred="pial",
            y_true="pial",
        ),
        curv=SurfaceSupervisedLoss(
            SymmetricCurvatureNormLoss(),
            y_pred="pial",
            y_true="pial",
        ),
    ),
)

head_weights = dict(svf=1.0, image=1.0, white=1.0, pial=1.0)

loss_weights = dict(
    svf=dict(gradient=1.0),
    image=dict(ncc=1.0),
    white=dict(chamfer=1.0, curv=1.0),
    pial=dict(chamfer=1.0, curv=1.0),
)

cfg_loss = LossParameters(functions, head_weights, loss_weights)
