import functools
import sys
import torch

from ignite.engine import Engine

import brainsynth
from brainsynth.transforms import EnsureDevice
from brainsynth.utilities import squeeze_nd

from brainnet.dict_utils import (
    recursively_apply_function,
    recursively_apply_method,
    recursive_dict_sum,
)
import brainnet.train.utilities
from brainnet import event_handlers
import brainnet.initializers
from brainnet.mesh.surface import load_deepsurfer_template, Surface


class Step:
    def __init__(
        self,
        preprocessor: brainsynth.Synthesizer | None,
        model: brainnet.BrainNet,
        template_resolution: int = 0,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.device = self.model.device
        self.ensure_device = EnsureDevice(self.device)

        self.template = load_deepsurfer_template(template_resolution, self.device)
        self.topologies = {h: s.topology for h, s in self.template.items()}

    def apply_affine(self, affine: torch.Tensor):
        """Apply an affine to the template."""
        return {k: v.apply_affine(affine) for k, v in self.template.items()}

    def vertex_dict_to_surface_dict(self, templates):
        return {
            (k0, k1): Surface(v, self.topologies[k0])
            for (k0, k1), v in templates.items()
        }

    def prepare_batch(
        self,
        images: dict[str, torch.Tensor],
        vox2ras: dict[str, torch.Tensor],
        y_true: dict[str, torch.Tensor] | None = None,
    ):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.

        The preprocessor needs to return the image and the associated
        voxel-to-ras transform.

        """
        if y_true is not None:
            y_true = self.ensure_device(y_true)

        if self.preprocessor is None:
            # assume preprocessor was applied when loading the data
            return images["image"], vox2ras["affine"], y_true
        else:
            # Remove batch dim, apply preprocessor, add batch dim back
            func = functools.partial(squeeze_nd, n=4, dim=0)
            images = recursively_apply_function(images, func)
            func = functools.partial(squeeze_nd, n=2, dim=0)
            vox2ras = recursively_apply_function(vox2ras, func)

            with torch.no_grad():
                out = self.preprocessor(images, affines=vox2ras, unpack=False)

            func = functools.partial(torch.unsqueeze, dim=0)
            out = recursively_apply_function(out, func)

            return out["image"], out["affine"], y_true

    def prepare_for_loss(self, affines):
        keys = zip(("lh", "lh", "rh", "rh"), ("lh", "brain", "rh", "brain"))
        return {
            (k0, k1): self.template[k0].apply_affine(affines[k1], return_surface=False)
            for k0, k1 in keys
        }

    def postprocess(self, y_pred):
        return y_pred


class TrainingStep(Step):
    def __init__(
        self,
        preprocessor: brainsynth.Synthesizer,
        model,
        criterion,
        optimizer: torch.optim.Optimizer,
        enable_amp: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(preprocessor, model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        if self.enable_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def compute_loss(self, y_pred, y_true):
        raw = self.criterion(y_pred, y_true)
        return dict(raw=raw, weighted=self.criterion.apply_weights(raw))

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        image, vox2ras, y_true = self.prepare_batch(*batch)

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            y_pred = self.model(image, vox2ras)

            y_pred = self.prepare_for_loss(y_pred)
            y_true = self.prepare_for_loss(y_true)
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
        loss = recursively_apply_method(loss, "item")

        # these are stored in engine.state.output
        return loss, image, vox2ras, y_pred, y_true


class EvaluationStep(Step):
    def __init__(self, preprocessor, model, criterion, enable_amp: bool = False):
        super().__init__(preprocessor, model)
        self.criterion = criterion
        self.enable_amp = enable_amp

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)

    def __call__(self, engine, batch):
        self.model.eval()

        image, vox2ras, y_true = self.prepare_batch(*batch)

        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                y_pred = self.model(image, vox2ras)

                y_pred = self.prepare_for_loss(y_pred)
                y_true = self.prepare_for_loss(y_true)
                loss = self.compute_loss(y_pred, y_true)

        loss = recursively_apply_method(loss, "item")

        y_pred = self.vertex_dict_to_surface_dict(y_pred)
        y_true = self.vertex_dict_to_surface_dict(y_true)

        return loss, image, vox2ras, y_pred, y_true


class PredictionStep(Step):
    def __init__(
        self,
        preprocessor,
        model,
        enable_amp: bool = False,
        **kwargs,
    ):
        super().__init__(preprocessor, model, **kwargs)
        self.enable_amp = enable_amp

    def postprocess(self, y_pred):
        return self.apply_affine(y_pred["brain"])

    def __call__(self, engine, batch):
        self.model.eval()

        image, vox2ras, _ = self.prepare_batch(*batch)

        with torch.inference_mode():
            with torch.autocast(self.model.device.type, enabled=self.enable_amp):
                y_pred = self.model(image, vox2ras)

        return y_pred


def write_example_prediction(y, out_dir):
    event_handlers.write_surfaces(y, out_dir, surf="template")


def write_example_event(
    engine: Engine,
    evaluators: dict[str, Engine],
    config: brainnet.config.ResultsParameters,
):
    vol_info = event_handlers.FREESURFER_VOLUME_INFO

    for prefix, e in evaluators.items():
        prefix = f"epoch-{engine.state.epoch:05d}.{prefix}"

        match e:
            case TrainingStep():
                _, image, vox2ras, y_pred, y_true = e.state.output
            case EvaluationStep():
                _, image, vox2ras, y_pred, y_true = e.state.output
            # case PredictionStep():
            #     y_pred = e.state.output
            case _:
                raise ValueError(f"Unknown engine type {type(e)}")

        vol_info["volume"] = tuple(image.shape[-3:])

        # write the volume that was used for prediction
        event_handlers.write_volume(
            image, vox2ras, config.examples_dir, prefix, label="image"
        )

        # write the predicted template positions
        for tag, y in zip(("pred", "true"), (y_pred, y_true)):
            # if config.examples_keys is None or label in config.examples_keys:
            y = {"-".join(k) if isinstance(k, tuple) else k: v for k, v in y.items()}
            event_handlers.write_surfaces(
                y, config.examples_dir, prefix, tag, surf="template", vol_info=vol_info
            )


def setup_model(setup):
    model = setup.model
    model.to(setup.device)
    return model


def create_trainer(setup, no_wandb: bool = False):
    # Overwrite args from command line if provided
    if no_wandb:
        setup.wandb.enable = False

    criterion = brainnet.initializers.init_criterion(setup.criterion)
    dataloader = brainnet.initializers.init_dataloader(setup.dataset, setup.dataloader)
    model = setup_model(setup)
    # model.compile()
    optimizer = brainnet.initializers.init_optimizer(setup.optimizer, model)
    synth = brainnet.initializers.init_synthesizer(setup.synthesizer)
    # for k,v in synth.items():
    #     if isinstance(v, torch.nn.Module):
    #         synth[k].compile()

    # =============================================================================
    # TRAINING
    # =============================================================================

    train_step = TrainingStep(
        synth["train"],
        model,
        criterion["train"],
        optimizer,
        setup.enable_amp,
        setup.trainer_gradient_accumulation_steps,
    )
    eval_step = EvaluationStep(
        synth["validation"],
        model,
        criterion["validation"],
        setup.enable_amp,
    )
    trainer = Engine(train_step)

    # The order in which the events are added to the engine is important!

    # Aggregate average loss over epoch
    brainnet.train.utilities.add_metric_to_engine(trainer)
    brainnet.train.utilities.add_terminal_logger(trainer)

    # Add evaluations
    evaluators = dict(
        validation=brainnet.train.utilities.add_evaluation_event(
            eval_step,
            engine=trainer,
            evaluate_on=setup.evaluator_evaluate_on,
            epoch_length=setup.evaluator_epoch_length,
            dataloader=dataloader["validation"],
            logger=event_handlers.MetricLogger(key="loss", name="validation"),
        ),
    )

    brainnet.train.utilities.add_wandb_logger(trainer, evaluators, setup.wandb)

    # Should be triggered after metrics has been computed!
    brainnet.train.utilities.add_custom_events(trainer, setup.trainer_events)
    for e in evaluators.values():
        brainnet.train.utilities.add_custom_events(e, setup.evaluator_events)

    # Include this in the checkpoint
    to_save = dict(
        model=model,
        optimizer=optimizer,
        engine=trainer,
        **{f"criterion[{k}]": v for k, v in criterion.items()},
    )
    if setup.enable_amp:
        to_save["grad_scaler"] = train_step.grad_scaler

    brainnet.train.utilities.add_model_checkpoint(trainer, to_save, setup.results)
    # brainnet.train.utilities.write_example_to_disk(
    #     trainer, evaluators, setup.results, event_handlers.write_input_image_with_affine
    # )
    brainnet.train.utilities.write_example_to_disk(
        trainer, evaluators, setup.results, write_example_event
    )
    brainnet.train.utilities.load_checkpoint_from_setup(to_save, setup)

    print("Setup completed.", end="\n\n")
    print(setup)

    return trainer, dataloader["train"]


if __name__ == "__main__":
    args = brainnet.train.utilities.argparser_topofit(sys.argv)
    brainnet.train.utilities.train(
        "alignment", args.specs, create_trainer, args.no_wandb
    )
