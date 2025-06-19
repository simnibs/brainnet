from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import optimizer_multiply_lr

loss_events = []
optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1001),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1501),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
]

events = loss_events + optimizer_events
