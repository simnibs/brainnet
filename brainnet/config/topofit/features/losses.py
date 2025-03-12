import torch

from brainnet.config.base import LossParameters
from brainnet.modules.loss_wrappers import SupervisedLoss
functions = dict(
    features = dict(
        decoder_3=SupervisedLoss(torch.nn.MSELoss(), "dec:3", "dec:3"),
        decoder_2=SupervisedLoss(torch.nn.MSELoss(), "dec:2", "dec:2"),
        decoder_1=SupervisedLoss(torch.nn.MSELoss(), "dec:1", "dec:1"),
        decoder_0=SupervisedLoss(torch.nn.MSELoss(), "dec:0", "dec:0"),
        # encoder_0=SupervisedLoss(torch.nn.MSELoss(), "encoder:0", "encoder:0"),
        # encoder_1=SupervisedLoss(torch.nn.MSELoss(), "encoder:1", "encoder:1"),
        # encoder_2=SupervisedLoss(torch.nn.MSELoss(), "encoder:2", "encoder:2"),
        # encoder_3=SupervisedLoss(torch.nn.MSELoss(), "encoder:3", "encoder:3"),
    ),
    # SR = dict(
    #     SR_1 = SupervisedLoss(torch.nn.MSELoss(), "sr1", "sr1"),
    # )
)

head_weights = dict(features=1.0)
# head_weights = dict(features=1.0, SR = 1.0)

loss_weights = dict(
    features=dict(
        decoder_3 = 1.0,
        decoder_2 = 1.0,
        decoder_1 = 1.0,
        decoder_0 = 1.0,
    ),
    # SR = dict(SR_1 = 5.0)
)

cfg_loss = LossParameters(functions, head_weights, loss_weights)
