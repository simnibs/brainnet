from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import (
    set_loss_weight,
    optimizer_multiply_lr,
    optimizer_reset,
)

loss_events = [
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=101),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             # ("white", "spring"): 10.0,
    #             # ("pial", "spring"): 10.0,
    #             ("white", "taubin"): 20.0,
    #             ("pial", "taubin"): 10.0,
    #             # ("white", "edge_var"): 5.0,
    #             # ("pial", "edge_var"): 2.5,
    #             # ("white", "tri_Q"): 2.5,
    #             # ("pial", "tri_Q"): 2.5,
    #         }
    #     ),
    # ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=201),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "negloglik"): 0.5,
                ("pial", "negloglik"): 0.5,
                ("white", "chamfer"): 0.0,
                ("pial", "chamfer"): 0.0,
                ("white", "edge_var"): 2.0,
                ("pial", "edge_var"): 1.0,
                ("white", "tri_quality"): 1.0,
                ("pial", "tri_quality"): 1.0,
                # ("white", "spring"): 5.0,
                # ("pial", "spring"): 5.0,
                # ("registration", "negloglik"): 0.05,
                # ("registration", "chamfer"): 0.0,
                ("registration", "chamfer"): 0.0,
                ("registration", "chamfer_w"): 1.0,
                # ("registration", "tri_quality"): 0.1,
            }
        ),
    ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=301),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             # ("white", "spring"): 2.0,  # 1.0,
    #             # ("pial", "spring"): 1.0,  # 0.5,
    #         }
    #     ),
    # ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=401),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                # ("white", "spring"): 1.0,  # 0.5,
                # ("pial", "spring"): 0.5,  # 0.25,
                ("white", "edge_var"): 1.0,
                ("pial", "edge_var"): 0.5,
                ("white", "tri_quality"): 0.5,
                ("pial", "tri_quality"): 0.5,
            }
        ),
    ),
]

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_STARTED(once=201),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=601),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
]

events = loss_events + optimizer_events
