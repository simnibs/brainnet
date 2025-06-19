from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import SupervisedLoss
from brainnet.modules.losses import NormLoss, MSNELoss, QuantileNormLoss

keys = [("lh", "lh"), ("rh", "rh"), ("lh", "brain"), ("rh", "brain")]
functions = {
    k: dict(
        MSE=SupervisedLoss(MSNELoss(), y_pred=k, y_true=k),
        distance=SupervisedLoss(NormLoss(), y_pred=k, y_true=k),
        distance_95=SupervisedLoss(QuantileNormLoss(0.95), y_pred=k, y_true=k),
    )
    for k in keys
}

head_weights_train = dict(zip(keys, (1.0, 1.0, 1.0, 1.0)))
head_weights_val = head_weights_train

loss_weights_train = {k: dict(MSE=0.0, distance=1.0, distance_95=0.0) for k in keys}
loss_weights_val = {k: dict(MSE=0.0, distance=1.0, distance_95=1.0) for k in keys}

train = LossParameters(functions, head_weights_train, loss_weights_train)
validation = LossParameters(functions, head_weights_val, loss_weights_val)
