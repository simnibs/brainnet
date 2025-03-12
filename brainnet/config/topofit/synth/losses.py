from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses_surface import (
    EdgeLengthVarianceLoss,
    FaceNormalConsistencyLoss,
    MatchedDistanceLoss,
    SampledSemiSymmetricL1Loss,
    SampledSemiSymmetricMSNormLoss,
    SampledSemiSymmetricSemiHardSNormLoss,
    SelfIntersectionCount,
    VertexToVertexAngleLoss,
)

# Surface loss:
# - loss is averaged across lh and rh (as available)


kw_white = dict(y_pred="white", y_true="white")
kw_pial = dict(y_pred="pial", y_true="pial")

functions = dict(
    white=dict(
        matched=SurfaceSupervisedLoss(
            MatchedDistanceLoss(),
            **kw_white,
        ),
        spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="white"),
        edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss("sampled_P", sym_weights=(0.5, 0.5)),
            **kw_white,
        ),
        hardchamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricSemiHardSNormLoss(
                "sampled_P", sym_weights=(0.5, 0.5), upper_split=0.25
            ),
            **kw_white,
        ),
        curv=SurfaceSupervisedLoss(
            SampledSemiSymmetricL1Loss("sampled_H"),
            **kw_white,
        ),
        sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="white"),
        # normals=SurfaceSupervisedLoss(
        #     SymmetricSampledMSELoss("normals_sampled"),
        #     **kw_white,
        # ),
    ),
    pial=dict(
        matched=SurfaceSupervisedLoss(
            MatchedDistanceLoss(),
            **kw_pial,
        ),
        spring=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="pial"),
        edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="pial"),
        chamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricMSNormLoss("sampled_P", sym_weights=(0.5, 0.5)),
            **kw_pial,
        ),
        hardchamfer=SurfaceSupervisedLoss(
            SampledSemiSymmetricSemiHardSNormLoss(
                "sampled_P", sym_weights=(0.5, 0.5), upper_split=0.25
            ),
            **kw_pial,
        ),
        curv=SurfaceSupervisedLoss(
            SampledSemiSymmetricL1Loss("sampled_H"),
            **kw_pial,
        ),
        sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="pial"),
        # normals=SurfaceSupervisedLoss(
        #     SymmetricSampledMSELoss("normals_sampled"),
        #     **kw_pial,
        # ),
    ),
    thickness=dict(
        angle=SurfaceRegularizationLoss(
            VertexToVertexAngleLoss(),  # cutoff=0.866
            y_pred=None,  # pass everything through, i.e., both white and pial
        ),
    ),
)

head_weights = dict(white=1.0, pial=1.0, thickness=1.0)

# loss_weights = dict(
#     white=dict(
#         matched=0.0,
#         spring=2.0,
#         edge=2.0,
#         chamfer=1.0,
#         hardchamfer=0.0,
#         curv=0.0,
#         sif=0.0,
#     ),
#     pial=dict(
#         matched=0.0,
#         spring=2.0,
#         edge=2.0,
#         chamfer=1.0,
#         hardchamfer=0.0,
#         curv=0.0,
#         sif=0.0,
#     ),
#     thickness=dict(angle=5.0),
# )
loss_weights = dict(
    white=dict(matched=0.0, spring=25.0, edge=10.0, chamfer=1.0, hardchamfer=0.0, curv=0.0, sif=0.0),
    pial=dict(matched=0.0, spring=25.0, edge=10.0, chamfer=1.0, hardchamfer=0.0, curv=0.0, sif=0.0),
    thickness=dict(angle = 5.0),
)

cfg_loss = LossParameters(functions, head_weights, loss_weights)
