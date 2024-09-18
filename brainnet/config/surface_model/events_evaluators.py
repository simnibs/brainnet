from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import set_loss_weight

 # NOTE
 #
 # Epoch/iteration numbers refer to those of the trainer engine.
 #
 # If an evaluator is not triggered on the event where an action is defined,
 # it will never be triggered, e.g., if evaluation is defined on every 20th
 # epoch, then once=1000 is valid whereas once=1005 is not; every=10 will only
 # be triggered half the time!

loss_events = [
    # Turn on losses
    config.EventAction(
        event=Events.EPOCH_COMPLETED(once=600),
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "chamfer"):   1.0, # new
            ("white", "curv"):      1.0, # new
            ("pial", "chamfer"):    1.0, # new
            ("pial", "curv"):       1.0, # new; white / 2
        }),
    ),
    # Remove losses
    config.EventAction(
        event=Events.EPOCH_COMPLETED(once=1400),
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "matched"):   0.0, # remove
            ("white", "hinge"):     0.0, # remove
            ("pial", "matched"):    0.0, # remove
            ("pial", "hinge"):      0.0, # remove
        }),
    ),

]

events = loss_events