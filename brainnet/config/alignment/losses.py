from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import SupervisedLoss
from brainnet.modules.losses import MSELoss, MSNELoss

functions = dict(
    lh_lh=dict(
        MSE=SupervisedLoss(MSNELoss(), y_pred="lh_lh", y_true="lh_lh"),
    ),
    rh_rh=dict(
        MSE=SupervisedLoss(MSNELoss(), y_pred="rh_rh", y_true="rh_rh"),
    ),
    lh_brain=dict(
        MSE=SupervisedLoss(MSNELoss(), y_pred="lh_brain", y_true="lh_brain"),
    ),
    rh_brain=dict(
        MSE=SupervisedLoss(MSNELoss(), y_pred="rh_brain", y_true="rh_brain"),
    ),
)

head_weights = dict(lh_lh=1.0, rh_rh=1.0, lh_brain=1.0, rh_brain=1.0)
loss_weights = dict(
    lh_lh=dict(MSE=1.0),
    rh_rh=dict(MSE=1.0),
    lh_brain=dict(MSE=1.0),
    rh_brain=dict(MSE=1.0),
)

train = LossParameters(functions, head_weights, loss_weights)
validation = train
