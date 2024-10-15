from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import set_head_weight, set_loss_weight, optimizer_set_lr, optimizer_multiply_lr

loss_events = [
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=210),
    #     handler=set_head_weight,
    #     kwargs=dict(weights=dict(white=1.0, pial=1.0)),
    # ),
    config.EventAction(
        event=Events.EPOCH_STARTED(once=410), # EPOCH_COMPLETED(once=400)
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "curv"): 1.0,
            ("pial", "curv"): 1.0,
            # ("seg", "dice"):
            # ("image", "ncc_2"): 0.0,
            # ("image", "ncc_4"): 0.0,
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