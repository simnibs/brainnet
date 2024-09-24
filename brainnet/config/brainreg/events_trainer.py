from ignite.engine import Events

from brainnet import config
from brainnet.event_handlers import set_loss_weight, optimizer_multiply_lr

loss_events = [
    # config.EventAction(
    #     event=Events.EPOCH_STARTED(once=701), # EPOCH_COMPLETED(once=400)
    #     handler=set_loss_weight,
    #     kwargs=dict(weights={("svf", "gradient"): 1.0}), # / 10
    # ),
]

# optimizer_events = [
#     config.EventAction(
#         event=Events.EPOCH_COMPLETED(once=1000),
#         handler=optimizer_multiply_lr,
#         kwargs=dict(factor=0.5),
#     ),
# ]

events = loss_events #+ optimizer_events