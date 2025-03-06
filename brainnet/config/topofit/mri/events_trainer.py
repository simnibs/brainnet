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
                # add
                # ("white", "curv"):      1.0,
                # ("pial", "curv"):       1.0,
            }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=201),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "chamfer"):       0.0,
                ("pial", "chamfer"):        0.0,
                ("white", "hardchamfer"):   1.0,
                ("pial", "hardchamfer"):    1.0,
            }
        ),
    ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=501),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "matched"):   0.01,   # / 100
    #             ("pial", "matched"):    0.01,   # / 100
    #             ("white", "spring"):     0.1,   # / 100
    #             ("pial", "spring"):      0.1,   # / 100
    #             ("white", "edge"):      1.0,    # / 5
    #             ("pial", "edge"):       1.0,    # / 5
    #             # new
    #             ("white", "chamfer"):   1.0,
    #             ("pial", "chamfer"):    1.0,
    #             ("white", "curv"):      1.0,
    #             ("pial", "curv"):       1.0,
    #         }
    #     ),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=751),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "matched"):   0.001,   # / 10
    #             ("pial", "matched"):    0.001,   # / 10
    #             ("white", "spring"):     0.01,    # / 10
    #             ("pial", "spring"):      0.01,    # / 10
    #             # ("white", "edge"):      1.0,    # / 5
    #             # ("pial", "edge"):       1.0,    # / 5
    #             # new
    #             # ("white", "curv"):      1.0,
    #             # ("pial", "curv"):       1.0,
    #         }
    #     ),
    # ),
    # Switch to resolution level 5
    # This causes curvature loss to increase approximately by a factor of 4 so
    # compensate for this in the weight
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1001),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "matched"):   0.0001,  # / 10
    #             ("pial", "matched"):    0.0001,  #
    #             # ("white", "spring"):     0.001,   # / 10
    #             # ("pial", "spring"):      0.001,

    #             # ("white", "edge"):      1.0,    # / 5
    #             # ("pial", "edge"):       1.0,    # / 5

    #             # Compensate for increased resolution
    #             # ("white", "curv"):     10.0,    # / 4
    #             # ("pial", "curv"):       5.0,    # / 4
    #         }
    #     ),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1251),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "matched"):   0.0001,    # / 10
    #             ("pial", "matched"):    0.0001,
    #             # ("white", "spring"):     0.0,    # / 10
    #             # ("pial", "spring"):      0.0,
    #             ("white", "edge"):      0.5,    # / 2
    #             ("pial", "edge"):       0.5,    # / 2

    #             ("white", "curv"):      0.5,    # / 2
    #             ("pial", "curv"):       0.5,    # / 2
    #         }
    #     ),
    # ),
    # Switch to resolution level 7
    # This causes curvature loss to increase approximately by a factor of 4 so
    # compensate for this in the weight
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1501),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             # Compensate for increased resolution
    #             ("white", "curv"):      2.5,     # / 4
    #             ("pial", "curv"):       1.25,    # / 4
    #         }
    #     ),
    # ),
]

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=51),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
]

events = loss_events + optimizer_events
