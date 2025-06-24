from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses_surface import (
    EdgeLengthVarianceLoss,
    TaubinLoss,
    MetricDistortionLoss,
    OrientedAreaLoss,
    SampledSemiSymmetricMSNormLoss,
    SampledSemiSymmetricNegLogLikLoss,
    SelfIntersectionCount,
    TriangleQualityLoss,
    VertexToVertexAngleLoss,
)

# Surface loss:
# - loss is averaged across left and right hemispheres (as available)

kw_white = dict(y_pred="white", y_true="white")
kw_pial = dict(y_pred="pial", y_true="pial")
kw_reg = dict(y_pred="registration", y_true="registration")

# kw_semisym = dict(weight_key="medial_wall", sym_weights=(0.5, 0.5))
# kw_semisym = dict(weight_key=None, sym_weights=(0.1, 0.9))

functions = dict(
    white=dict(
        # spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="white"),
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss(
                value_key=("interpolated", "points"),
            ),
            **kw_white,
        ),
        edge_var=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        negloglik=SurfaceSupervisedLoss(
            SampledSemiSymmetricNegLogLikLoss(
                value_key=("interpolated", "points"), weight_key="sigma"
            ),
            **kw_white,
        ),
        sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="white"),
        tri_quality=SurfaceRegularizationLoss(TriangleQualityLoss(), y_pred="white"),
        taubin=SurfaceRegularizationLoss(TaubinLoss(), y_pred="white"),
        # curv=SurfaceSupervisedLoss(
        #     SampledSemiSymmetricNL1Loss(value_key=("interpolated", "H")),
        #     # SampledSemiSymmetricNSELoss("H", **kw_semisym),
        #     **kw_white,
        # ),
    ),
    pial=dict(
        # spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="pial"),
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss(value_key=("interpolated", "points")),
            **kw_pial,
        ),
        edge_var=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="pial"),
        negloglik=SurfaceSupervisedLoss(
            SampledSemiSymmetricNegLogLikLoss(
                value_key=("interpolated", "points"), weight_key="sigma"
            ),
            **kw_pial,
        ),
        sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="pial"),
        taubin=SurfaceRegularizationLoss(TaubinLoss(), y_pred="pial"),
        tri_quality=SurfaceRegularizationLoss(TriangleQualityLoss(), y_pred="pial"),
        # smoothness_vertex_chamfer = SurfaceSupervisedLoss(IndexedSmoothnessLoss(),**kw_pial),
        # curv=SurfaceSupervisedLoss(
        #     SampledSemiSymmetricNL1Loss(value_key=("interpolated", "H")),
        #     # SampledSemiSymmetricNSELoss("H", **kw_semisym),
        #     **kw_pial,
        # ),
    ),
    thickness=dict(
        angle=SurfaceRegularizationLoss(
            VertexToVertexAngleLoss(),  # cutoff=0.866
            y_pred=None,  # pass everything through, i.e., both white and pial
        ),
    ),
    # The spherical coordinates are stored as vertex data on the white surface
    registration=dict(
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss(value_key=("interpolated", "points")),
            **kw_reg,
            # weight_key="sigma",
        ),
        chamfer_w=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss(
                value_key=("interpolated", "points"),
                weight_key="sigma",
                weight_invert=True,
                weight_norm_dim=-1,
            ),
            **kw_reg,
        ),
        # tri_quality=SurfaceRegularizationLoss(
        #     TriangleQualityLoss(), y_pred="registration"
        # ),
        area=SurfaceRegularizationLoss(OrientedAreaLoss(), y_pred="registration"),
        distortion=SurfaceRegularizationLoss(
            MetricDistortionLoss(), y_pred="registration"
        ),
    ),
)

head_weights = dict(white=1.0, pial=1.0, thickness=1.0, registration=0.01)

# fmt: off
loss_weights = dict(
    white = dict(
        chamfer     =    1.0,
        negloglik   =    0.0,
        sif         =    0.0,
        # spring      =   40.0,
        taubin      =   40.0,
        edge_var    =    5.0,
        tri_quality =    2.5,
    ),
    pial = dict(
        chamfer     =    1.0,
        negloglik   =    0.0,
        sif         =    0.0,
        # spring      =   40.0,
        taubin      =   20.0,
        edge_var    =    2.5,
        tri_quality =    2.5,
    ),
    thickness = dict(
        angle       =    1.0,
    ),
    registration = dict(
        chamfer     =    1.0,
        chamfer_w   =    0.0,
        area        =   10.0, # 100.0
        distortion  =   10.0,
    ),
)
# fmt: on

train = LossParameters(functions, head_weights, loss_weights)

# loss_weights | dict(white=dict(sif=1.0), pial=dict(sif=1.0)
validation = LossParameters(functions, head_weights, loss_weights)
