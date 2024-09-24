from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import set_head_weight, set_loss_weight, optimizer_multiply_lr

loss_events = [
    config.EventAction(
        event=Events.EPOCH_COMPLETED(once=200),
        handler=set_loss_weight,
        kwargs=dict(weights={("white", "hinge"): 10.0}), # / 10
    ),
    config.EventAction(
        event=Events.EPOCH_COMPLETED(once=600),
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "matched"):   0.01,   # /= 100
            ("white", "hinge"):     0.1,    # /= 100
            ("white", "chamfer"):   1.0,    # new
            ("white", "curv"):     50.0,    # new
            ("pial", "matched"):    0.01,
            ("pial", "hinge"):      0.1,
            ("pial", "chamfer"):    1.0,    # new
            ("pial", "curv"):      50.0,    # new
        }),
    ),
    config.EventAction(
        event=Events.EPOCH_COMPLETED(once=1000),
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "matched"):   0.001,  # / 10
            ("white", "hinge"):     0.01,   # / 10
            ("white", "curv"):     10.0,    # / 5
            ("pial", "matched"):    0.001,  # / 10
            ("pial", "hinge"):      0.01,   # / 10
            ("pial", "curv"):      10.0,    # / 5
        }),
    ),
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=1001),
    #     handler=set_loss_weight,
    #     kwargs=dict(weights={
    #         ("white", "matched"):   0.001,  # / 10
    #         ("white", "hinge"):     0.01,   # / 10
    #         ("white", "chamfer"):   1.0,    # new
    #         ("white", "curv"):     10.0,    # / 5
    #         ("pial", "matched"):    0.001,  # / 10
    #         ("pial", "hinge"):      0.01,   # / 10
    #         ("pial", "chamfer"):    1.0,    # new
    #         ("pial", "curv"):       5.0,    # / 5
    #     }),
    # ),
    config.EventAction(
        event=Events.EPOCH_COMPLETED(once=1400),
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "matched"):   0.0, # remove
            ("white", "hinge"):     0.0, # remove
            ("white","edge"):       2.5,
            ("pial", "matched"):    0.0, # remove
            ("pial", "hinge"):      0.0, # remove
            ("pial","edge"):        2.5,
        }),
    ),
    config.EventAction(
        event=Events.EPOCH_COMPLETED(once=1800),
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white","edge"):       1.0,
            ("pial","edge"):        1.0,
            ("pial", "curv"):       2.5, # white / 4
        }),
    ),
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
        event=Events.EPOCH_COMPLETED(once=1000),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
]

events = loss_events + optimizer_events