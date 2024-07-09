from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import set_loss_weight, optimizer_multiply_lr

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
            ("pial", "curv"):      25.0,    # new; white / 2
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
            ("pial", "curv"):       5.0,    # / 5
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
]

optimizer_events = [
    config.EventAction(
        event=Events.EPOCH_COMPLETED(once=1000),
        handler=optimizer_multiply_lr,
        kwargs=dict(factor=0.5),
    ),
]

events = loss_events + optimizer_events