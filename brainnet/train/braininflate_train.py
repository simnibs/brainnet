import copy
import importlib
import sys
import torch

from ignite.engine import Engine

from brainsynth.transforms import EnsureDevice

import brainnet.config
import brainnet.train.utilities
from brainnet import event_handlers
import brainnet.initializers
from brainnet.utilities import recursive_dict_sum, recursive_itemize


class SupervisedStep:
    def __init__(
        self,
        model: brainnet.BrainNet,
        criterion: brainnet.Criterion,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = self.model.device
        self.ensure_device = EnsureDevice(self.device)

        surfaces = ["inflated"]

        self.template_surfaces = {}
        for h,surf_obj in self.model.surface.items():
            self.template_surfaces[h] = {}
            for s in surfaces:
                self.template_surfaces[h][s] = copy.deepcopy(surf_obj)

    def as_surface(self, data, surface_name="inflated"):
        """Insert vertices data from `data` into template and replace `data`
        with the template.
        """
        for h, surfaces in data.items():
            for s,v in surfaces.items():
                self.template_surfaces[h][s].vertices = v
                data[h][s] = self.template_surfaces[h][s]

    def prepare_batch(self, batch):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        _, y_true, init_verts = batch
        y_true = self.ensure_device(y_true)
        self.as_surface(y_true)
        init_verts = self.ensure_device(init_verts)

        return dict(surface=y_true), init_verts

        # if self.synthesizer is None:
        #     # assume synthesizer was applied when loading the data
        #     return batch
        # else:
        #     images, surfaces, init_verts = batch

        #     # REMOVE BATCH DIMENSION
        #     for k, v in images.items():
        #         images[k] = v.squeeze(0)
        #     for k, v in surfaces.items():
        #         for kk, vv in v.items():
        #             surfaces[k][kk] = vv.squeeze(0)
        #     for k, v in init_verts.items():
        #         init_verts[k] = v.squeeze(0)

        #     with torch.no_grad():
        #         y_true = self.synthesizer(images, surfaces, init_verts, unpack=False)

        #     image = y_true.pop("image")
        #     init_verts = y_true.pop("initial_vertices")

        #     # ADD BATCH DIMENSION
        #     image = image.unsqueeze(0)
        #     for k, v in y_true.items():
        #         if k == "surface":
        #             for k2, v2 in v.items():
        #                 for k3, v3 in v2.items():
        #                     y_true[k][k2][k3] = v3.unsqueeze(0)
        #         else:
        #             y_true[k] = v.unsqueeze(0)
        #     for k, v in init_verts.items():
        #         init_verts[k] = v.unsqueeze(0)

        #     return image, y_true, init_verts

    # def prepare_loss(self, y_pred, y_true):
    #     if (k := "surface") in y_pred:
    #         self.update_surface_template(self.surface_template["y_pred"], y_pred[k])
    #         self.update_surface_template(self.surface_template["y_true"], y_true[k])

    #         self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

    def compute_loss(self, y_pred, y_true):
        raw = self.criterion(y_pred, y_true)
        return dict(raw=raw, weighted=self.criterion.apply_weights(raw))


class SupervisedTrainingStep(SupervisedStep):
    def __init__(
        self, model, criterion, optimizer,
        gradient_accumulation_steps: int = 1,
        enable_amp: bool = False
    ) -> None:
        super().__init__(model, criterion)
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.enable_amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        # no image
        y_true, init_verts = self.prepare_batch(batch)

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            inf = self.model(init_verts)
            y_pred = dict(surface={})
            for h, v in inf.items():
                y_pred["surface"][h] = dict(inflated=v)

            loss = self.compute_loss(y_pred, y_true)
            total_loss = recursive_dict_sum(loss["weighted"])
            total_loss /= self.gradient_accumulation_steps

        # exit if loss diverges
        if total_loss > 1e6 or torch.isnan(total_loss):
            raise RuntimeError(f"Loss diverged (loss = {total_loss})")

        if self.enable_amp:
            self.grad_scaler.scale(total_loss).backward()
            if engine.state.iteration % self.gradient_accumulation_steps == 0:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss.backward()  # backpropagate loss
            if engine.state.iteration % self.gradient_accumulation_steps == 0:
                self.optimizer.step()  # update parameters
                # Reset gradients in optimizer. Otherwise gradients would
                # accumulate across multiple passes (whenever .backward is
                # called)
                self.optimizer.zero_grad()
        loss = recursive_itemize(loss)

        # these are stored in engine.state.output
        return loss, None, y_pred, y_true

class EvaluationStep(SupervisedStep):
    def __init__(self, model, criterion, enable_amp: bool = False):
        super().__init__(model, criterion)
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        self.model.eval()

        _, y_true, init_verts = self.prepare_batch(batch)

        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                y_pred = self.model(init_verts)
                loss = self.compute_loss(y_pred, y_true)

        # we don't need the weighted loss
        loss = recursive_itemize(loss["raw"])

        return loss, None, y_pred, y_true


def train(args):

    """

    train_setup_file = "brainnet.config.braininflate.main"
    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")
    train_setup.wandb.enable = False

    """

    train_setup_file = args.config  # "brainnet.config.cortex.main"

    print("Setting up training...")

    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")

    sep_line = 79 * "="

    # Overwrite args from command line if provided
    if args.load_checkpoint is not None:
        train_setup.train_params.load_checkpoint = args.load_checkpoint
    if args.max_epochs is not None:
        train_setup.train_params.max_epochs = args.max_epochs
    if args.no_wandb:
        train_setup.wandb.enable = False

    criterion = brainnet.initializers.init_criterion(train_setup.criterion)
    dataloader = brainnet.initializers.init_dataloader(
        train_setup.dataset, train_setup.dataloader
    )
    model = train_setup.model # already initialize
    optimizer = brainnet.initializers.init_optimizer(train_setup.optimizer, model)

    # =============================================================================
    # TRAINING
    # =============================================================================

    train_step = SupervisedTrainingStep(
        model,
        criterion["train"],
        optimizer,
        train_setup.train_params.gradient_accumulation_steps,
        enable_amp=train_setup.train_params.enable_amp,
    )
    trainer = Engine(train_step)

    # The order in which the events are added to the engine is important!

    # Aggregate average loss over epoch
    brainnet.train.utilities.add_metric_to_engine(trainer)
    brainnet.train.utilities.add_terminal_logger(trainer)

    # Add evaluation
    kwargs = dict(
        engine=trainer,
        evaluate_on=train_setup.train_params.evaluate_on,
        epoch_length=train_setup.train_params.epoch_length_val,
    )

    evaluators = dict(
        # train = brainnet.train.utilities.add_evaluation_event(
        #     EvaluationStep(
        #         synth["train"],
        #         model,
        #         criterion["validation"], # NOTE here we use the validation criterion!
        #         train_setup.train_params.enable_amp,
        #     ),
        #     dataloader=dataloader["train"],
        #     logger=event_handlers.MetricLogger(key="loss", name="train"),
        #     **kwargs,
        # ),
        validation=brainnet.train.utilities.add_evaluation_event(
            EvaluationStep(
                model,
                criterion["validation"],
                train_setup.train_params.enable_amp,
            ),
            dataloader=dataloader["validation"],
            logger=event_handlers.MetricLogger(key="loss", name="validation"),
            **kwargs,
        ),
    )

    brainnet.train.utilities.add_wandb_logger(trainer, evaluators, train_setup.wandb)

    # Should be triggered after metrics has been computed!
    brainnet.train.utilities.add_custom_events(
        trainer, train_setup.train_params.events_trainer
    )
    for e in evaluators.values():
        brainnet.train.utilities.add_custom_events(
            e, train_setup.train_params.events_evaluators
        )

    # Include this in the checkpoint
    to_save = dict(
        model=model,
        optimizer=optimizer,
        engine=trainer,
        **{f"criterion[{k}]": v for k, v in criterion.items()},
    )
    if train_setup.train_params.enable_amp:
        to_save["grad_scaler"] = train_step.grad_scaler

    brainnet.train.utilities.add_model_checkpoint(trainer, to_save, train_setup.results)
    brainnet.train.utilities.write_example_to_disk(trainer, evaluators, train_setup.results)
    brainnet.train.utilities.load_checkpoint(to_save, train_setup)

    print("Setup completed. Starting training at epoch ...")

    print(sep_line)
    print(f"Config file     {train_setup_file}")
    print(f"Project         {train_setup.project:30s}")
    print(f"Run             {train_setup.run:30s}")
    print(f"Output dir      {train_setup.results.out_dir}")
    print(f"Wandb enabled   {train_setup.wandb.enable}")
    print(sep_line)

    # Start the training
    epoch_length = train_setup.train_params.epoch_length_train or len(iter(dataloader["train"]))
    # trainer.state.epoch_length = epoch_length
    trainer.run(
        dataloader["train"],
        epoch_length=epoch_length,
        max_epochs=train_setup.train_params.max_epochs,
    )


if __name__ == "__main__":
    args = brainnet.train.utilities.parse_args(sys.argv)
    train(args)
