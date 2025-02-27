import torch

from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import SupervisedLoss
functions = dict(
    features = dict(
        decoder_3=SupervisedLoss(torch.nn.MSELoss(), "decoder:3", "decoder:3"),
        # decoder_2=SupervisedLoss(torch.nn.MSELoss(), "decoder:2", "decoder:2"),
        # decoder_1=SupervisedLoss(torch.nn.MSELoss(), "decoder:1", "decoder:1"),
        # encoder_0=SupervisedLoss(torch.nn.MSELoss(), "encoder:0", "encoder:0"),
        # encoder_1=SupervisedLoss(torch.nn.MSELoss(), "encoder:1", "encoder:1"),
        # encoder_2=SupervisedLoss(torch.nn.MSELoss(), "encoder:2", "encoder:2"),
        # encoder_3=SupervisedLoss(torch.nn.MSELoss(), "encoder:3", "encoder:3"),
    ),
)

head_weights = dict(features=1.0)
loss_weights = dict(
    features=dict(
        decoder_3 = 1.0,
    ),
)

cfg_loss = LossParameters(functions, head_weights, loss_weights)
