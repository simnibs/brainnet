import functools
import importlib
import sys
import torch

from ignite.engine import Engine

import brainsynth
# from brainsynth.utilities import apply_affine

from brainnet.dict_utils import (
    recursively_apply_function,
    recursively_apply_method,
    recursive_dict_sum,
)
import brainnet.helpers.utils
from brainnet import event_handlers
import brainnet.initializers

from brainnet.config.topofit.features import train_parameters


class SupervisedStep:
    def __init__(
        self,
        synthesizer: None | brainsynth.Synthesizer,
        model: torch.nn.Module,
        criterion: brainnet.Criterion,
        # subdivision: int,
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
            images, affines, _, init_verts = batch

            # Remove batch dim
            func = functools.partial(torch.squeeze, dim=0)
            images = recursively_apply_function(images, func)
            init_verts = recursively_apply_function(init_verts, func)

            with torch.no_grad():
                y_true = self.synthesizer(
                    images, initial_vertices=init_verts, affines=affines, unpack=False
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
        freeze_body: bool = False,
    ) -> None:
        super().__init__(synthesizer, model, criterion)
        self.pretrained_model = pretrained_model
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        self.freeze_body = freeze_body
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


def setup_model(setup):
    return brainnet.initializers.init_model(setup.model)


def setup_pretrained_model(setup):
    return brainnet.initializers.init_model(setup.pretrained_model)


def create_trainer(setup, no_wandb: bool = False):
    # Overwrite args from command line if provided
    if no_wandb:
        setup.wandb.enable = False

    criterion = brainnet.initializers.init_criterion(setup.criterion)
    dataloader = brainnet.initializers.init_dataloader(setup.dataset, setup.dataloader)
    model = setup_model(setup)
    pretrained_model = setup_pretrained_model(setup)
    # model.compile()
    optimizer = brainnet.initializers.init_optimizer(setup.optimizer, model)
    synth = brainnet.initializers.init_synthesizer(setup.synthesizer)
    # for k,v in synth.items():
    #     if isinstance(v, torch.nn.Module):
    #         synth[k].compile()

    # =============================================================================
    # TRAINING
    # =============================================================================

    train_step = SupervisedTrainingStep(
        synth["train"],
        pretrained_model,
        model,
        criterion["train"],
        optimizer,
        setup.trainer_gradient_accumulation_steps,
        setup.enable_amp,
        setup.UNET_FREEZE,
    )
    eval_step = EvaluationStep(
        synth["validation"],
        pretrained_model,
        model,
        criterion["validation"],
        setup.enable_amp,
    )
    trainer = Engine(train_step)

    # # Set medial wall weights
    # if md_weights is not None:
    #     n_vertices = train_step.surface_template["y_pred"]["lh"]["white"].topology.n_vertices
    #     md_weights = torch.tensor(md_weights, device=model.device)
    #     medial_wall = brainnet.loss_weights.MedialWall(md_weights, n_vertices)

    #     for h,v in train_step.surface_template["y_true"].items():
    #         w = medial_wall.weights[h]
    #         for s in v:
    #             train_step.surface_template["y_true"][h][s].vertex_data["medial_wall"] = w
    #             train_step.surface_template["y_pred"][h][s].vertex_data["medial_wall"] = w

    #             eval_step.surface_template["y_true"][h][s].vertex_data["medial_wall"] = w
    #             eval_step.surface_template["y_pred"][h][s].vertex_data["medial_wall"] = w

    # The order in which the events are added to the engine is important!

    # Aggregate average loss over epoch
    brainnet.helpers.utils.add_metric_to_engine(trainer)
    brainnet.helpers.utils.add_terminal_logger(trainer)

    # Add evaluations

    evaluators = dict(
        # train = brainnet.helpers.utils.add_evaluation_event(
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
        validation=brainnet.helpers.utils.add_evaluation_event(
            eval_step,
            engine=trainer,
            evaluate_on=setup.evaluator_evaluate_on,
            epoch_length=setup.evaluator_epoch_length,
            dataloader=dataloader["validation"],
            logger=event_handlers.MetricLogger(key="loss", name="validation"),
        ),
    )

    brainnet.helpers.utils.add_wandb_logger(trainer, evaluators, setup.wandb)

    # Should be triggered after metrics has been computed!
    brainnet.helpers.utils.add_custom_events(trainer, setup.trainer_events)
    for e in evaluators.values():
        brainnet.helpers.utils.add_custom_events(e, setup.evaluator_events)

    # Include this in the checkpoint
    to_save = dict(
        model=model,
        optimizer=optimizer,
        engine=trainer,
        **{f"criterion[{k}]": v for k, v in criterion.items()},
    )
    if setup.enable_amp:
        to_save["grad_scaler"] = train_step.grad_scaler

    brainnet.helpers.utils.add_model_checkpoint(trainer, to_save, setup.results)
    brainnet.helpers.utils.write_example_to_disk(trainer, evaluators, setup.results)
    brainnet.helpers.utils.load_checkpoint_from_setup(to_save, setup)

    brainnet.helpers.utils.load_checkpoint(
        dict(model=pretrained_model),
        setup.pretrained_checkpoint_filename,
        setup.device,
    )

    print("Setup completed.", end="\n\n")
    print(setup)

    return trainer, dataloader["train"]


def train(args):
    """

    python brainnet/train/topofit.py synth_1mm --no-wandb

    args = brainnet.helpers.utils.argparser_topofit(
        "brainnet/train/topofit.py t1w_1mm --no-wandb".split()
    )

    """

    print(f"Using training specs: {args.specs}", end="\n\n")

    specs = importlib.import_module(
        f".{args.specs}", "brainnet.config.topofit.features"
    )

    try:
        PHASES = dict(OVERRIDE=getattr(specs, "OVERRIDE"))
        PHASES = specs.PHASES if PHASES is None else PHASES
    except AttributeError:
        PHASES = specs.PHASES

    for name, phase in PHASES.items():
        print(f"STARTING TRAINING PHASE: {name}")
        print(79 * "=")
        print(f"Specification\n    {phase}")
        print(f"Defaults\n    {specs.DEFAULTS}")

        setup = train_parameters.TrainParameters(**(specs.DEFAULTS | phase))
        trainer, dataloader = create_trainer(setup, args.no_wandb)
        trainer.run(
            dataloader,
            epoch_length=setup.trainer_epoch_length or len(iter(dataloader)),
            max_epochs=setup.max_epochs,
        )
        print(f"TRAINING PHASE DONE: {name}", end="\n\n")
        print(79 * "=")


if __name__ == "__main__":
    args = brainnet.helpers.utils.argparser_topofit(sys.argv)
    train(args)
