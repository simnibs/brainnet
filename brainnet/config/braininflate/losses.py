from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import (
    SurfaceRegularizationLoss,
    SurfaceSupervisedLoss,
)
from brainnet.modules.losses_surface import (
    SurfaceRMSELoss,
    MatchedDistanceLoss,
    SurfaceEdgeRMSELoss,
    SphericalNormalLoss,
    CentralAngleLoss,
)

kw_sphere = dict(y_pred="sphere.reg", y_true="sphere.reg")

functions = {
    "sphere.reg": dict(
        # distance=SurfaceSupervisedLoss(SurfaceRMSELoss(), **kw_sphere),
        distance=SurfaceSupervisedLoss(CentralAngleLoss(), **kw_sphere),
        edge=SurfaceSupervisedLoss(SurfaceEdgeRMSELoss(), **kw_sphere),
        normal=SurfaceRegularizationLoss(SphericalNormalLoss(), y_pred=kw_sphere["y_pred"]),

        # distance=SurfaceSupervisedLoss(SphericalArcLoss(radius=100.0), **kw_sphere),
        # distance=SurfaceSupervisedLoss(SphericalAxisAlignedArcLoss(), **kw_sphere),
        # edge=SurfaceSupervisedLoss(SphericalEdgeLoss(), **kw_sphere),
        # normal=SurfaceRegularizationLoss(SphericalNormalLoss(), y_pred=kw_sphere["y_pred"]),
        #curv=SurfaceSupervisedLoss(MatchedCurvatureMSELoss(), **kw_inflated),
        #smoothness=SurfaceRegularizationLoss(SmoothnessLoss(), y_pred="inflated"),
        #spring=SurfaceSupervisedLoss(SpringForceLoss(), **kw_inflated),
        #curvvar=SurfaceRegularizationLoss(MeanCurvatureVarianceLoss(), y_pred="inflated"),
        # face_normal=SurfaceRegularizationLoss(FaceNormalConsistencyLoss(), y_pred="inflated"),
        # edge=SurfaceRegularizationLoss(EdgeLengthVarianceLoss(), y_pred="white"),
        #sif=SurfaceRegularizationLoss(SelfIntersectionCount(), y_pred="inflated"),
    ),
}

head_weights = {"sphere.reg": 1.0}
loss_weights = {"sphere.reg": dict(distance=1.0, edge=10.0, normal=1.0)}

cfg_loss = LossParameters(functions, head_weights, loss_weights)
