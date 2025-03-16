from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
    optimizer_reset,
)

loss_events = [
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=21),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "curv"):  1.0,
    #             ("pial", "curv"):   1.0,
    #         }
    #     ),
    # ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=51),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "spring"):    5.0,  # / 5
                ("pial", "spring"):     5.0,  # / 5
                ("white", "edge"):      2.0,  # / 5
                ("pial", "edge"):       2.0,  # / 5
            }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=101),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "spring"):    1.0,   # / 5
                ("pial", "spring"):     1.0,   # / 5
                ("white", "edge"):      1.0,   # / 2
                ("pial", "edge"):       1.0,   # / 2
            }
        ),
    ),

    config.EventAction(
        event=Events.EPOCH_STARTED(once=301),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "spring"):    0.5,   # / 2
                ("pial", "spring"):     0.5,   # / 2
                # ("white", "chamfer"):       0.0,
                # ("pial", "chamfer"):        0.0,
                # ("white", "hardchamfer"):   1.0,
                # ("pial", "hardchamfer"):    1.0,
            }
        ),
    ),

    config.EventAction(
        event=Events.EPOCH_STARTED(once=501),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "curv"):      1.0,
                # ("pial", "curv"):       1.0,
            }
        ),
    ),

]

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=51),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
]

events = loss_events + optimizer_events
