# trainer = monai.engines.Trainer()

# validator = monai.engines.Evaluator(
#     device=,
#     val_data_loader=,
#     # epoch_length=,

# )
# validation_handler = monai.handler.ValidationHandler(
#     interval = ,
#     validator = validator,

# )

# validation_handler.attach(trainer)


# evaluator = monai.engines.Validator(
#     device=,
#     val_data_loader=,
#     # epoch_length=,

# )

# monai.inferers.SimpleInferer()

# handler = monai.handlers.CheckpointLoader(
#     load_path="/test/checkpoint.pt",
#     load_dict=save_dict,
#     map_location=map_location,
#     strict=True
# )
# handler(trainer)


# saver = monai.handlers.CheckpointSaver()
# saver.attach(engine)

# monai.handlers.MeanSquaredError()


# from monai.metrics import LossMetric

# dice_metric = LossMetric(DiceLoss)
from monai.engines import Trainer

trainer = Engine(train_step)

def train_step(engine, batch):

    ds_id, images, surfaces, info = batch

    ds_id = ds_id[0] # ds_id is a tuple of len 1
    # self.to_device(image)
    # self.to_device(surface)
    # self.to_device(info)


    with torch.no_grad():

        data = self.synthesizer(images, surfaces, info)

        if "synth" not in data:
            # select a random contrast from the list of alternative images
            avail = getattr(self.config.dataset.alternative_synth, ds_id)
            sel = torch.randint(
                0, len(avail), (1,), device=self.synthesizer.device
            )
            data["synth"] = data[avail[sel]]

        # do it AFTER synth for now...
        self.to_device(data)

        image = data.pop("synth")
        y_true = data

    t1 = perf_counter()

    loss, wloss = self.step(image, y_true, ds_id)

    t2 = perf_counter()

    # compute weighted loss
    # wloss = self.criterion.apply_weights(loss)
    wloss_sum = brainnet.utilities.recursive_dict_sum(wloss)

    # Reset gradients in optimizer. Otherwise gradients would
    # accumulate across multiple passes
    self.optimizer.zero_grad()
    # Compute and accumulate gradients. backward() frees
    # intermediate values of the graph (e.g., activations)
    wloss_sum.backward()
    # Update parameters (i.e., gradients)
    self.optimizer.step()

    # log the loss
    epoch.loss_update(loss, wloss)

    self.hook_on_step_end(step)

    t3 = perf_counter()


    synth_time = t1-t0
    model_time = t2-t1
    step_time = t3-t0

    print("synth time", synth_time)
    print("model time", model_time)
    print("step  time", step_time)

