from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
)

loss_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=201),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "hinge"): 10.0,   # / 10
                ("pial", "hinge"):  10.0,   # / 10
            }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=501),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "matched"):   0.01,   # / 100
                ("pial", "matched"):    0.01,   # / 100
                ("white", "hinge"):     0.1,    # / 100
                ("pial", "hinge"):      0.1,    # / 100
                # new
                ("white", "chamfer"):   1.0,
                ("pial", "chamfer"):    1.0,
                ("white", "curv"):     40.0,
                ("pial", "curv"):      20.0,
            }
        ),
    ),
    # Switch to resolution level 5
    # This causes curvature loss to increase approximately by a factor of 4 so
    # compensate for this in the weight
    config.EventAction(
        event=Events.EPOCH_STARTED(once=801),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "matched"):   0.001,  # / 10
                ("pial", "matched"):    0.001,  #

                ("white", "hinge"):     0.01,   # / 10
                ("pial", "hinge"):      0.01,

                ("white", "edge"):      2.0,    # / 2
                ("pial", "edge"):       2.0,    # / 2

                # Compensate for increased resolution
                ("white", "curv"):     10.0,    # / 4
                ("pial", "curv"):       5.0,    # / 4
            }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1101),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "matched"):   0.0,    # / 10
                ("white", "hinge"):     0.0,    # / 10
                ("pial", "matched"):    0.0,
                ("pial", "hinge"):      0.0,

                ("white", "edge"):      1.0,    # / 2
                ("pial", "edge"):       1.0,    # / 2
            }
        ),
    ),
    # Switch to resolution level 6
    # This causes curvature loss to increase approximately by a factor of 4 so
    # compensate for this in the weight
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1401),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                # Compensate for increased resolution
                ("white", "curv"):      2.5,     # / 4
                ("pial", "curv"):       1.25,    # / 4
            }
        ),
    ),
]

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=801),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1401),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
]

events = loss_events + optimizer_events
