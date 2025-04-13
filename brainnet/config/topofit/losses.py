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
        # matched=SurfaceSupervisedLoss(MatchedDistanceLoss(n_vertices=62), **kw_white),
        spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="white"),
        edge_var=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        tri_Q=SurfaceRegularizationLoss(TriangleQualityLoss(), y_pred="white"),
        taubin=SurfaceRegularizationLoss(TaubinLoss(), y_pred="white"),
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss(
                value_key=("interpolated", "points"), sym_weights=(0.5, 0.5),
            ),
            **kw_white,
        ),
        # normal=SurfaceSupervisedLoss(
        #     SampledSemiSymmetricCosSimLoss(
        #         value_key=("interpolated", "normal"),
        #     ),
        #     **kw_white,
        # ),
        # curv=SurfaceSupervisedLoss(
        #     SampledSemiSymmetricNL1Loss(value_key=("interpolated", "H")),
        #     # SampledSemiSymmetricNSELoss("H", **kw_semisym),
        #     **kw_white,
        # ),
        sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="white"),
    ),
    pial=dict(
        # matched=SurfaceSupervisedLoss(MatchedDistanceLoss(n_vertices=62), **kw_pial),
        spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="pial"),
        taubin=SurfaceRegularizationLoss(TaubinLoss(), y_pred="pial"),
        edge_var=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="pial"),
        tri_Q=SurfaceRegularizationLoss(TriangleQualityLoss(), y_pred="pial"),
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss(
                value_key=("interpolated", "points"), sym_weights=(0.5, 0.5),
            ),
            **kw_pial,
        ),
        # normal=SurfaceSupervisedLoss(
        #     SampledSemiSymmetricCosSimLoss(
        #         value_key=("interpolated", "normal"),
        #     ),
        #     **kw_pial,
        # ),
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
)

head_weights = dict(white=1.0, pial=1.0, thickness=1.0)

# fmt: off
loss_weights = dict(
    white = dict(
        chamfer  =    1.0,
        # curv     =    0.0,
        sif      =    0.0,
        # spring   =   40.0,
        taubin   =  100.0,
        edge_var =    5.0, # 10
        tri_Q  =      2.5, # 5
    ),
    pial = dict(
        chamfer  =    1.0,
        # curv     =    0.0,
        sif      =    0.0,
        # spring   =   40.0,
        taubin   =  100.0,
        edge_var =    2.5, # 5
        tri_Q  =      2.5, # 5
    ),
    thickness = dict(
        angle    =    1.0,
    ),
)
# fmt: on

cfg_loss = LossParameters(functions, head_weights, loss_weights)
