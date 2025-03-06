from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
)

loss_events = [
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
                ("white", "spring"):    1.0,  # / 5
                ("pial", "spring"):     1.0,  # / 5
                ("white", "edge"):      1.0,  # / 5
                ("pial", "edge"):       1.0,  # / 5
            }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=201),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                # ("white", "chamfer"):       0.0,
                # ("pial", "chamfer"):        0.0,
                # ("white", "hardchamfer"):   1.0,
                # ("pial", "hardchamfer"):    1.0,
                ("white", "curv"):       1.0,
                ("pial", "curv"):        1.0,
                ("white", "spring"):    0.1,  # / 5
                ("pial", "spring"):     0.1,  # / 5
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
