from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
)

loss_events = [
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1201),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("sphere.reg", "edge"): 50.0,   # / 10
    #         }
    #     ),
    # ),
]

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=401),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1401),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.5),
    # ),
]

events = loss_events + optimizer_events
