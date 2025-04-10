from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
    optimizer_reset,
)

loss_events = [

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=51),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "spring"):    10.0,  # / 5
    #             ("white", "edge"):      50.0,  # / 2
    #         }
    #     ),
    # ),

]

optimizer_events = [

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=201),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.5),
    # ),

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=401),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.5),
    # ),

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=301),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.2),
    # ),

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=501),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.5),
    # ),
]

events = loss_events + optimizer_events
