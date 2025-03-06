from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
    optimizer_reset,
)

loss_events = []

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=401),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=751),
    #     handler=optimizer_reset,
    # ),
]

events = loss_events + optimizer_events
