from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses_surface import (
    EdgeLengthVarianceLoss,
    TaubinLoss,
    FaceNormalConsistencyLoss,
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

# kw_semisym = dict(weight_key="medial_wall", sym_weights=(0.5, 0.5))
# kw_semisym = dict(weight_key=None, sym_weights=(0.1, 0.9))

functions = dict(
    white=dict(
        spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="white"),
        edge_var=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        tri_Q=SurfaceRegularizationLoss(TriangleQualityLoss(), y_pred="white"),
        taubin=SurfaceRegularizationLoss(TaubinLoss(), y_pred="white"),
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss(
                value_key=("interpolated", "points"),
            ),
            **kw_white,
        ),
        negloglik=SurfaceSupervisedLoss(
            SampledSemiSymmetricNegLogLikLoss(
                value_key=("interpolated", "points"),
                weight_key="sigma",
            ),
            **kw_white,
        ),
        # curv=SurfaceSupervisedLoss(
        #     SampledSemiSymmetricNL1Loss(value_key=("interpolated", "H")),
        #     # SampledSemiSymmetricNSELoss("H", **kw_semisym),
        #     **kw_white,
        # ),
        sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="white"),
    ),
    pial=dict(
        spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="pial"),
        taubin=SurfaceRegularizationLoss(TaubinLoss(), y_pred="pial"),
        edge_var=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="pial"),
        tri_Q=SurfaceRegularizationLoss(TriangleQualityLoss(), y_pred="pial"),
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss(
                value_key=("interpolated", "points"),
            ),
            **kw_pial,
        ),
        negloglik=SurfaceSupervisedLoss(
            SampledSemiSymmetricNegLogLikLoss(
                value_key=("interpolated", "points"),
                weight_key="sigma",
            ),
            **kw_pial,
        ),
        # smoothness_vertex_chamfer = SurfaceSupervisedLoss(IndexedSmoothnessLoss(),**kw_pial),
        # curv=SurfaceSupervisedLoss(
        #     SampledSemiSymmetricNL1Loss(value_key=("interpolated", "H")),
        #     # SampledSemiSymmetricNSELoss("H", **kw_semisym),
        #     **kw_pial,
        # ),
        sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="pial"),
    ),
    thickness=dict(
        angle=SurfaceRegularizationLoss(
            VertexToVertexAngleLoss(),  # cutoff=0.866
            y_pred=None,  # pass everything through, i.e., both white and pial
        ),
    ),
    # The spherical coordinates are stored as vertex data on the white surface
    # sphere=dict(
    #     arc=SurfaceSupervisedLoss(
    #         SampledSemiSymmetricMSNormLoss(value_key=("interpolated", "sphere")),
    #         **kw_white,
    #     ),
    #     # tri_Q=
    # ),
)

head_weights = dict(white=1.0, pial=1.0, thickness=1.0)

# fmt: off
loss_weights = dict(
    white = dict(
        chamfer     =    1.0,
        negloglik   =    0.0,
        sif         =    0.0,
        # spring      =   40.0,
        taubin      =   40.0,
        edge_var    =    5.0,
        tri_Q       =    2.5,
    ),
    pial = dict(
        chamfer     =    1.0,
        negloglik   =    0.0,
        sif         =    0.0,
        # spring      =   40.0,
        taubin      =   20.0,
        edge_var    =    2.5,
        tri_Q       =    2.5,
    ),
    thickness = dict(
        angle       =    1.0,
    ),
)
# fmt: on

train = LossParameters(functions, head_weights, loss_weights)

# loss_weights | dict(white=dict(sif=1.0), pial=dict(sif=1.0)
validation = LossParameters(functions, head_weights, loss_weights)
