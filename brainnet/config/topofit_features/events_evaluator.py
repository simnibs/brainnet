from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
    optimizer_reset,
)

loss_events = []

optimizer_events = []

events = loss_events + optimizer_events
