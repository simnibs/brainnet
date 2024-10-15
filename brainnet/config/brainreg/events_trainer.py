from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import set_head_weight, set_loss_weight, optimizer_set_lr, optimizer_multiply_lr

loss_events = [
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=121), # EPOCH_COMPLETED(once=400)
    #     handler=set_head_weight,
    #     kwargs=dict(weights=dict(seg=5.0)),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=121), # EPOCH_COMPLETED(once=400)
    #     handler=set_loss_weight,
    #     kwargs=dict(weights={
    #         ("white", "hinge"): 25.0,
    #         ("pial", "hinge"): 25.0,
    #         # ("seg", "dice"):
    #         # ("image", "ncc_2"): 0.0,
    #         # ("image", "ncc_4"): 0.0,
    #     }
    #     ),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=141), # EPOCH_COMPLETED(once=400)
    #     handler=set_loss_weight,
    #     kwargs=dict(weights={
    #         ("svf", "gradient"): 50,
    #         ("white", "hinge"): 10.0,
    #         ("pial", "hinge"): 10.0,
    #     }
    #     ),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=281),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "hinge"): 5.0,   # / 10
    #             ("pial", "hinge"):  5.0,   # / 10
    #         }
    #     ),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=341),
    #     handler=set_loss_weight,
    #     kwargs=dict(
    #         weights={
    #             ("white", "curv"): 20.0,   # / 10
    #             ("pial", "curv"):  10.0,   # / 10
    #         }
    #     ),
    # ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=401),
        handler=set_loss_weight,
        kwargs=dict(
            weights={
                ("white", "hinge"): 2.5,   # / 10
                ("pial", "hinge"):  2.5,   # / 10
            }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=801), # EPOCH_COMPLETED(once=400)
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "chamfer"): 0.75,
            ("pial", "chamfer"): 0.75,
        }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=901), # EPOCH_COMPLETED(once=400)
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "chamfer"): 1.5,
            ("pial", "chamfer"): 1.5,
        }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1001), # EPOCH_COMPLETED(once=400)
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "chamfer"): 2.0,
            ("white", "hinge"): 5.0,
            ("pial", "chamfer"): 2.0,
            ("pial", "hinge"): 5.0,
        }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1101), # EPOCH_COMPLETED(once=400)
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "chamfer"): 4.0,
            ("white", "hinge"): 10.0,
            ("pial", "chamfer"): 4.0,
            ("pial", "hinge"): 10.0,
        }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1201), # EPOCH_COMPLETED(once=400)
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "chamfer"): 5.0,
            # ("white", "hinge"): 20.0,
            ("pial", "chamfer"): 5.0,
            # ("pial", "hinge"): 20.0,
        }
        ),
    ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=1301), # EPOCH_COMPLETED(once=400)
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "chamfer"): 10.0,
            # ("white", "hinge"): 20.0,
            ("pial", "chamfer"): 10.0,
            # ("pial", "hinge"): 20.0,
        }
        ),
    ),

]

optimizer_events = [
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1301),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.5),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=801),
    #     handler=optimizer_multiply_lr,
    #     kwargs=dict(factor=0.5),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1001),
    #     handler=optimizer_set_lr,
    #     kwargs=dict(lr=5e-5),
    # ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1301),
    #     handler=optimizer_set_lr,
    #     kwargs=dict(lr=1e-3),
    # ),
]


events = loss_events + optimizer_events