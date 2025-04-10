import torch

from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import SupervisedLoss
functions = dict(
    features = dict(
        decoder_3=SupervisedLoss(torch.nn.MSELoss(), "dec:3", "dec:3"),
        decoder_2=SupervisedLoss(torch.nn.MSELoss(), "dec:2", "dec:2"),
        decoder_1=SupervisedLoss(torch.nn.MSELoss(), "dec:1", "dec:1"),
        decoder_0=SupervisedLoss(torch.nn.MSELoss(), "dec:0", "dec:0"),
    ),
)

head_weights = dict(features=1.0)

loss_weights = dict(
    features=dict(
        decoder_3 = 8.0,
        decoder_2 = 4.0,
        decoder_1 = 2.0,
        decoder_0 = 1.0,
    ),
)

cfg_loss = LossParameters(functions, head_weights, loss_weights)
