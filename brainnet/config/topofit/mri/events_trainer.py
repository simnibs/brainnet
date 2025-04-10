from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
    optimizer_reset,
)

loss_events = [

    config.EventAction(
        event=Events.EPOCH_STARTED(once=101),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                # ("white", "spring"): 10.0,
                # ("pial", "spring"):  10.0,
                ("white", "taubin"):   20.0,
                ("pial", "taubin"):    20.0,

                ("white", "edge_var"):  5.0,
                ("pial", "edge_var"):   2.5,
                ("white", "tri_Q"):     2.5,
                ("pial", "tri_Q"):      2.5,
            }
        ),
    ),

    config.EventAction(
        event=Events.EPOCH_STARTED(once=301),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                # ("white", "spring"): 1.0,
                # ("pial", "spring"):  0.5,
                ("white", "taubin"):   5.0,
                ("pial", "taubin"):    2.5,

                # ("white", "edge_var"):  2.5,
                # ("pial", "edge_var"):   1.25,
                # ("white", "tri_Q"):     1.25,
                # ("pial", "tri_Q"):      1.25,
                ("white", "edge_var"):  1.0,
                ("pial", "edge_var"):   0.5,
                ("white", "tri_Q"):     0.5,
                ("pial", "tri_Q"):      0.5,
            }
        ),
    ),

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=401),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             # ("white", "spring"): 0.5,
    #             # ("pial", "spring"):  0.25,

    #             ("white", "edge_var"):  1.0,
    #             ("pial", "edge_var"):   0.5,
    #             ("white", "tri_Q"):     0.5,
    #             ("pial", "tri_Q"):      0.5,

    #         }
    #     ),
    # ),


    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=501),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "matched"):     0.001,
    #             ("pial", "matched"):      0.001,

    #             ("white", "spring"):      0.5,
    #             ("pial", "spring"):       0.5,

    #             ("white", "edge_var"):    5.0,
    #             ("pial", "edge_var"):     2.5,

    #         }
    #     ),
    # ),

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=601),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "matched"):    0.0001,
    #             ("pial", "matched"):     0.0001,

    #             ("pial", "spring"):      0.25,
    #         }
    #     ),
    # ),

]

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=201),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=501),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.5),
    # ),

    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=501),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.5),
    # ),
]

events = loss_events + optimizer_events
