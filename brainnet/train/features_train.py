import copy
import functools
import importlib
import sys
import torch

from ignite.engine import Engine

import brainsynth

from brainnet.dict_utils import recursively_apply_function
import brainnet.train.utilities
from brainnet import event_handlers
import brainnet.initializers
from brainnet.dict_utils import recursive_dict_sum, recursively_apply_method


class SupervisedStep:
    def __init__(
        self,
        synthesizer: None | brainsynth.Synthesizer,
        model: brainnet.BrainNet,
        criterion: brainnet.Criterion,
    ) -> None:
        self.synthesizer = synthesizer
        self.model = model
        self.criterion = criterion
        self.device = self.model.device

    def prepare_batch(self, batch):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        if self.synthesizer is None:
            # assume synthesizer was applied when loading the data
            return batch
        else:
            images, _, init_verts = batch

            # Remove batch dim
            func = functools.partial(torch.squeeze, dim=0)
            images = recursively_apply_function(images, func)
            init_verts = recursively_apply_function(init_verts, func)

            with torch.no_grad():
                y_true = self.synthesizer(
                    images, initial_vertices=init_verts, unpack=False
                )

            # Add batch dim
            func = functools.partial(torch.unsqueeze, dim=0)
            y_true = recursively_apply_function(y_true, func)

            del y_true["surface"]
            del y_true["initial_vertices"]

            return y_true

    def compute_loss(self, y_pred, y_true):
        raw = self.criterion(y_pred, y_true)
        return dict(raw=raw, weighted=self.criterion.apply_weights(raw))


class SupervisedTrainingStep(SupervisedStep):
    def __init__(
        self,
        synthesizer,
        pretrained_model,
        model,
        criterion,
        optimizer,
        gradient_accumulation_steps: int = 1,
        enable_amp: bool = False,
    ) -> None:
        super().__init__(synthesizer, model, criterion)
        self.pretrained_model = pretrained_model
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.enable_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def __call__(self, engine, batch) -> tuple:
        self.model.train()
        self.pretrained_model.eval()

        batch = self.prepare_batch(batch)

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            y_pred = self.model.body(batch["image"])
            with torch.no_grad():
                y_true = self.pretrained_model.body(batch["t1w"])

            # y_true["sr1"] = batch["image_hires"]
            # y_pred["sr1"] = self.model.heads["sr1"](y_pred)

            mask = batch["brain_dist_map"] < 10.0

            # a little inaccurate but does not matter so much as it is just
            # for weighting the losses
            subsamp = [2 ** (3 - int(k.split(":")[-1])) for k in y_true if "dec:" in k]
            mask = {
                k: mask[..., ::s, ::s, ::s].ravel() for k, s in zip(y_true, subsamp)
            }
            # mask["sr1"] = mask["dec:3"]

            y_pred_masked = {
                k: v.reshape(*v.shape[:2], -1)[..., mask[k]] for k, v in y_pred.items()
            }
            y_true_masked = {
                k: v.reshape(*v.shape[:2], -1)[..., mask[k]] for k, v in y_true.items()
            }

            loss = self.compute_loss(y_pred_masked, y_true_masked)
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
        loss = recursively_apply_method(loss, "item")

        # these are stored in engine.state.output
        return loss, batch["image"], y_pred, y_true


class EvaluationStep(SupervisedStep):
    def __init__(
        self, synthesizer, pretrained_model, model, criterion, enable_amp: bool = False
    ):
        super().__init__(synthesizer, model, criterion)
        self.pretrained_model = pretrained_model
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        self.model.eval()
        self.pretrained_model.eval()

        batch = self.prepare_batch(batch)

        with torch.autocast(self.device.type, enabled=self.enable_amp):
            with torch.inference_mode():
                y_true = self.pretrained_model.body(batch["t1w"])
                y_pred = self.model.body(batch["image"])

                # y_true["sr1"] = batch["image_hires"]
                # y_pred["sr1"] = self.model.heads["sr1"](y_pred)

                mask = batch["brain_dist_map"] < 10.0

                # a little inaccurate but does not matter so much as it is just
                # for weighting the losses
                subsamp = [
                    2 ** (3 - int(k.split(":")[-1])) for k in y_true if "dec:" in k
                ]
                mask = {
                    k: mask[..., ::s, ::s, ::s].ravel() for k, s in zip(y_true, subsamp)
                }
                # mask["sr1"] = mask["dec:3"]

                y_pred_masked = {
                    k: v.reshape(*v.shape[:2], -1)[..., mask[k]]
                    for k, v in y_pred.items()
                }
                y_true_masked = {
                    k: v.reshape(*v.shape[:2], -1)[..., mask[k]]
                    for k, v in y_true.items()
                }

                loss = self.compute_loss(y_pred_masked, y_true_masked)

        # we don't need the weighted loss
        loss = recursively_apply_method(loss["raw"], "item")

        return loss, batch["image"], y_pred, y_true


def train(args):
    """

    train_setup_file = "brainnet.config.topofit.mri.main"
    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")
    train_setup.wandb.enable = False

    args = brainnet.train.utilities.parse_args(
        "brainnet/train/features_train.py brainnet.config.topofit.features.main --load-checkpoint 200 --max-epochs 250 --no-wandb".split()
    )

    """

    train_setup_file = args.config  # "brainnet.config.cortex.main"

    print("Setting up training...")

    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")
    cfg_pretrained_model = getattr(
        importlib.import_module(train_setup_file), "cfg_pretrained_model"
    )
    ckpt_pretrained_model = getattr(
        importlib.import_module(train_setup_file), "ckpt_pretrained_model"
    )

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
    pretrained_model = brainnet.initializers.init_model(
        copy.deepcopy(cfg_pretrained_model)
    )
    # pretrained_model_opt = torch.compile(pretrained_model)
    model = brainnet.initializers.init_model(train_setup.model)
    # model_opt = torch.compile(model)
    optimizer = brainnet.initializers.init_optimizer(train_setup.optimizer, model)
    synth = brainnet.initializers.init_synthesizer(train_setup.synthesizer)
    # synth_opt = {}
    # for k,v in synth.items():
    #     if isinstance(v, torch.nn.Module):
    #         synth_opt[k] = torch.compile(v)

    # =============================================================================
    # TRAINING
    # =============================================================================

    train_step = SupervisedTrainingStep(
        synth["train"],
        pretrained_model,
        model,
        criterion["train"],
        optimizer,
        train_setup.train_params.gradient_accumulation_steps,
        enable_amp=train_setup.train_params.enable_amp,
    )
    trainer = Engine(train_step)

    # Set medial wall weights
    # False = 0 = non-MD
    # True = 1 = MD
    # weights = torch.tensor([1.0, 0.25], device=model.device)
    # medial_wall_weights = WeightsMedialWall(weights).get_weights()
    # medial_wall_weights = medial_wall_weights[
    #     :train_step.surface_template["y_true"]["lh"]["white"].topology.n_vertices
    # ][None]
    # criterion["train"].set_weights_medial_wall(medial_wall_weights)
    # criterion["validation"].set_weights_medial_wall(medial_wall_weights)

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
                synth["validation"],
                pretrained_model,
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
    brainnet.train.utilities.write_example_to_disk(
        trainer, evaluators, train_setup.results
    )
    brainnet.train.utilities.load_checkpoint_from_setup(to_save, train_setup)
    brainnet.train.utilities.load_checkpoint(
        dict(model=pretrained_model), ckpt_pretrained_model, train_setup.device
    )
    print("Setup completed. Starting training at epoch ...")

    print(sep_line)
    print(f"Config file     {train_setup_file}")
    print(f"Project         {train_setup.project:30s}")
    print(f"Run             {train_setup.run:30s}")
    print(f"Output dir      {train_setup.results.out_dir}")
    print(f"Wandb enabled   {train_setup.wandb.enable}")
    print(sep_line)

    # Start the training
    epoch_length = train_setup.train_params.epoch_length_train or len(
        iter(dataloader["train"])
    )
    # trainer.state.epoch_length = epoch_length
    trainer.run(
        dataloader["train"],
        epoch_length=epoch_length,
        max_epochs=train_setup.train_params.max_epochs,
    )


if __name__ == "__main__":
    args = brainnet.train.utilities.parse_args(sys.argv)
    train(args)
