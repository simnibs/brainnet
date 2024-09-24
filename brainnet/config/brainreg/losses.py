from brainnet.config.base import LossParameters

from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    RegularizationLoss,
    SupervisedLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses import (
    DiceFromLabelsLoss,
    EdgeLengthVarianceLoss,
    HingeLoss,
    MatchedDistanceLoss,
    GradientLoss,
    MSELoss,
    NormalizedCrossCorrelationLoss,
    ScaledNormalizedCrossCorrelationLoss,
    SymmetricChamferLoss,
    SymmetricCurvatureNormLoss,
)

ncc_wrap_kwargs = dict(
    y_pred="t1w_areg_mni",
    y_true="t1w_areg_mni",
    y_pred_valid="t1w_areg_mni_mask",
    y_true_valid="t1w_areg_mni_mask",
)

patch_size = (100, 100, 100)
spatial_size = (192, 224, 192)

scale2 = 0.5
scale4 = 0.25

functions = dict(
    svf=dict(gradient=RegularizationLoss(GradientLoss(device="cuda"), y_pred="svf")),
    image=dict(
        ncc_1=SupervisedLoss(
            NormalizedCrossCorrelationLoss(
                stride=2,
                use_patch=True,
                patch_size=(100, 100, 100),
                spatial_size=spatial_size,
                device="cuda",
            ),
            **ncc_wrap_kwargs
        ),
        ncc_2=SupervisedLoss(
            ScaledNormalizedCrossCorrelationLoss(
                interp_factor=scale2,
                stride=2,
                use_patch=True,
                patch_size=[int(scale2 * p) for p in patch_size],
                spatial_size=[int(scale2 * s) for s in spatial_size],
                device="cuda",
            ),
            **ncc_wrap_kwargs
        ),
        ncc_4=SupervisedLoss(
            ScaledNormalizedCrossCorrelationLoss(interp_factor=scale4, device="cuda"),
            **ncc_wrap_kwargs
        ),
        # mse=SupervisedLoss(MSELoss(), y_pred="t1w_areg_mni", y_true="t1w_areg_mni")
    ),
    seg=dict(
        dice=SupervisedLoss(
            DiceFromLabelsLoss(num_classes=56 + 1),
            y_pred="brainseg_with_extracerebral",
            y_true="brainseg_with_extracerebral",
            # y_pred_valid="t1w_areg_mni_mask",
            # y_true_valid="t1w_areg_mni_mask",
        ),
    ),
    white=dict(
        chamfer=SurfaceSupervisedLoss(
            SymmetricChamferLoss(), y_pred="white", y_true="white"
        ),
        # curv=SurfaceSupervisedLoss(
        #     SymmetricCurvatureNormLoss(),
        #     y_pred="white",
        #     y_true="white",
        # ),
        edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        hinge=SurfaceRegularizationLoss(HingeLoss(), y_pred="white"),
        # matched=SurfaceSupervisedLoss(
        #     MatchedDistanceLoss(), y_pred="white", y_true="white",
        # ),
    ),
    pial=dict(
        chamfer=SurfaceSupervisedLoss(
            SymmetricChamferLoss(),
            y_pred="pial",
            y_true="pial",
        ),
        # curv=SurfaceSupervisedLoss(
        #     SymmetricCurvatureNormLoss(),
        #     y_pred="pial",
        #     y_true="pial",
        # ),
        edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="pial"),
        hinge=SurfaceRegularizationLoss(HingeLoss(), y_pred="pial"),
        # matched=SurfaceSupervisedLoss(
        #     MatchedDistanceLoss(), y_pred="pial", y_true="pial",
        # ),
    ),
)

head_weights = dict(
    svf=1.0,
    image=1.0,
    seg=1.0,
    # white=0.1,
    # pial=0.1,
)

loss_weights = dict(
    svf=dict(gradient=1.0), # 2.5 -> 5
    image=dict(ncc_1=1.0/3.0, ncc_2=1.0/3.0, ncc_4=1.0/3.0),
    seg = dict(dice=1.0),
    # image = dict(mse=1.0),
    # white=dict(chamfer=1.0, edge=1.0, hinge=1.0),
    # pial=dict(chamfer=1.0, edge=1.0, hinge=1.0),
    # white=dict(edge=1.0, hinge=1.0, matched=0.1),
    # pial=dict(edge=1.0, hinge=1.0, matched=0.1),

)

cfg_loss = LossParameters(functions, head_weights, loss_weights)
