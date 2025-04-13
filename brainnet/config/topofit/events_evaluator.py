from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import set_loss_weight

 # NOTE
 #
 # Epoch/iteration numbers refer to those of the trainer engine!
 #
 # If an evaluator is not triggered on the event where an action is defined,
 # it will never be triggered, e.g., if evaluation is defined on every 20th
 # epoch, then once=1000 is valid whereas once=1005 is not; every=10 will only
 # be triggered half the time!

loss_events = [
    # Turn on losses
    config.EventAction(
        event=Events.EPOCH_STARTED(once=110),
        handler=set_loss_weight,
        kwargs=dict(weights={
            ("white", "sif"):       1.0,
            ("pial", "sif"):        1.0,
        }),
    ),
]

events = loss_events