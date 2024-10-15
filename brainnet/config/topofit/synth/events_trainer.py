from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_head_weight,
    set_loss_weight,
    optimizer_multiply_lr,
)


# train_setup.train_params.gradient_accumulation_steps = 2
# train_setup.dataloader["train"].

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
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=401),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "hinge"): 1.0,    # / 10
    #             ("pial", "hinge"):  1.0,    # / 10
    #         }
    #     ),
    # ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=501),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "matched"):   0.01,   # / 100
                ("white", "hinge"):     0.1,    # / 100
                ("pial", "matched"):    0.01,   # / 100
                ("pial", "hinge"):      0.1,    # / 100
                # new
                ("white", "chamfer"):   1.0,
                ("white", "curv"):     40.0,
                ("pial", "chamfer"):    1.0,
                ("pial", "curv"):      20.0,
            }
        ),
    ),
    # Switch to resolution level 5
    # This causes curvature loss to increase approximately by a factor 2-3 so
    # compensate for this in the weight
    config.EventAction(
        event=Events.EPOCH_STARTED(once=801),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "matched"):   0.001,  # / 10
                ("white", "hinge"):     0.01,   # / 10
                ("pial", "matched"):    0.001,
                ("pial", "hinge"):      0.01,

                ("white", "chamfer"):   1.0,
                ("white", "curv"):     10.0, # 50
                ("pial", "chamfer"):    1.0,
                ("pial", "curv"):       5.0, # 25

                # ("white", "curv"):     10.0,    # / 5
                # ("pial", "curv"):       5.0,    # / 5

                ("white", "edge"):      1.0,    # / 2
                ("pial", "edge"):       1.0,    # / 2
            }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1201),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "matched"):   0.0,  # / 10
                ("white", "hinge"):     0.0,   # / 10
                ("pial", "matched"):    0.0,
                ("pial", "hinge"):      0.0,

                # ("white", "curv"):     10.0,    # / 5
                # ("pial", "curv"):       5.0,    # / 5
            }
        ),
    ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1401),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "curv"):     2.5,     # / 4
    #             ("pial", "curv"):      1.25,    # / 4
    #         }
    #     ),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1001),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             # ("white", "matched"):   0.001,  # / 10
    #             # ("white", "hinge"):     0.01,   # / 10
    #             # ("pial", "matched"):    0.001,
    #             # ("pial", "hinge"):      0.01,


    #             ("white", "edge"):      2.5,    # / 2
    #             ("pial", "edge"):       2.5,    # / 2
    #         }
    #     ),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1401),
    #     handler=set_loss_weight,
    #     kwargs=dict(weights={
    #         ("white", "matched"):   0.0,
    #         ("white", "hinge"):     0.0,
    #         ("pial", "matched"):    0.0,
    #         ("pial", "hinge"):      0.0,


    #     }),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=3001),
    #     handler=set_head_weight,
    #     kwargs=dict(weights={
    #         "thickness":  1.0,
    #     }),
    # ),
]

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1201),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
]

events = loss_events + optimizer_events
