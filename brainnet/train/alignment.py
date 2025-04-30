import copy
import functools
import importlib
import sys
import torch

from ignite.engine import Engine

import brainsynth
from brainsynth.transforms import EnsureDevice

from brainnet.dict_utils import (
    recursively_apply_function,
    recursively_apply_method,
    recursive_dict_sum,
)
import brainnet.train.utilities
from brainnet import event_handlers
import brainnet.initializers
from brainnet.config.alignment import train_parameters
from brainnet.mesh.surface import Surface
from brainnet.mesh.topology import DeepSurferTopology


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
        self.ensure_device = EnsureDevice(self.device)

        t_lh = DeepSurferTopology.recursive_subdivision(5, device=self.device)[-1]
        t_rh = copy.deepcopy(t_lh)
        t_rh.reverse_face_orientation()

        self.topologies = dict(lh=t_lh, rh=t_rh)

        self.template = dict(
            lh=Surface(
                brainsynth.load_cortical_template("lh", self.device)["vertices"][
                    : t_lh.n_vertices
                ],
                t_lh,
            ),
            rh=Surface(
                brainsynth.load_cortical_template("rh", self.device)["vertices"][
                    : t_rh.n_vertices
                ],
                t_rh,
            ),
        )

    def apply_affines_to_template(self, affines):
        return dict(
            lh=dict(
                lh=Surface(
                    self.template["lh"].apply_affine(affines["lh"]),
                    self.template["lh"].faces,
                ),
                brain=Surface(
                    self.template["lh"].apply_affine(affines["brain"]),
                    self.template["lh"].faces,
                ),
            ),
            rh=dict(
                rh=Surface(
                    self.template["rh"].apply_affine(affines["rh"]),
                    self.template["rh"].faces,
                ),
                brain=Surface(
                    self.template["rh"].apply_affine(affines["brain"]),
                    self.template["rh"].faces,
                ),
            ),
        )

    def apply_affines_to_template1(self, affines):
        # return dict(
        #     lh=dict(
        #         lh=self.template["lh"].apply_affine(affines["lh"]),
        #         brain=self.template["lh"].apply_affine(affines["brain"]),
        #     ),
        #     rh=dict(
        #         rh=self.template["rh"].apply_affine(affines["rh"]),
        #         brain=self.template["rh"].apply_affine(affines["brain"]),
        #     ),
        # )
        return dict(
            lh_lh=self.template["lh"].apply_affine(affines["lh"]),
            lh_brain=self.template["lh"].apply_affine(affines["brain"]),
            rh_rh=self.template["rh"].apply_affine(affines["rh"]),
            rh_brain=self.template["rh"].apply_affine(affines["brain"]),
        )

    def vertex_dict_to_surface_dict(self, templates):
        return {k: Surface(v, self.topologies[k[:2]]) for k, v in templates.items()}

    def prepare_batch(self, batch):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        if self.synthesizer is None:
            # assume synthesizer was applied when loading the data
            return batch[0]["t1w"], batch[1]["t1w"], batch[2]
        else:
            images, vox2mri, y_true = batch
            y_true = self.ensure_device(y_true)

            # Remove batch dim
            func = functools.partial(torch.squeeze, dim=0)
            images = recursively_apply_function(images, func)
            vox2mri = recursively_apply_function(vox2mri, func)

            with torch.no_grad():
                out = self.synthesizer(images, affines=vox2mri, unpack=False)

            # Add batch dim
            func = functools.partial(torch.unsqueeze, dim=0)
            out = recursively_apply_function(out, func)

            image = out.pop("image")
            vox2mri = out.pop("affine")

            return image, out, vox2mri, y_true

    def compute_loss(self, y_pred, y_true):
        raw = self.criterion(y_pred, y_true)
        return dict(raw=raw, weighted=self.criterion.apply_weights(raw))


class SupervisedTrainingStep(SupervisedStep):
    def __init__(
        self,
        synthesizer,
        model,
        criterion,
        optimizer,
        gradient_accumulation_steps: int = 1,
        enable_amp: bool = False,
    ) -> None:
        super().__init__(synthesizer, model, criterion)
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_amp = enable_amp
        if self.enable_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        image, images, vox2mri, y_true = self.prepare_batch(batch)

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            y_pred = self.model(image, vox2mri)

            y_pred = self.apply_affines_to_template1(y_pred)
            y_true = self.apply_affines_to_template1(y_true)

            loss = self.compute_loss(y_pred, y_true)

            # loss = self.compute_loss(
            #     recursively_apply_method(y_pred, "ravel"),
            #     recursively_apply_method(y_true, "ravel"),
            # )
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

        # y_pred = dict(surface=self.apply_affines_to_template(y_pred))
        # y_true = dict(surface=self.apply_affines_to_template(y_true))

        # these are stored in engine.state.output
        return loss, image, vox2mri, y_pred, y_true


class EvaluationStep(SupervisedStep):
    def __init__(self, synthesizer, model, criterion, enable_amp: bool = False):
        super().__init__(synthesizer, model, criterion)
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        self.model.eval()

        image, images, vox2mri, y_true = self.prepare_batch(batch)

        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                y_pred = self.model(image, vox2mri)

                y_pred = self.apply_affines_to_template1(y_pred)
                y_true = self.apply_affines_to_template1(y_true)

                loss = self.compute_loss(y_pred, y_true)

        # we don't need the weighted loss
        loss = recursively_apply_method(loss["raw"], "item")

        y_pred = self.vertex_dict_to_surface_dict(y_pred)
        y_true = self.vertex_dict_to_surface_dict(y_true)

        return loss, image, vox2mri, y_pred, y_true


def setup_model(setup):
    model = setup.model
    model.to(setup.device)
    return model


def create_trainer(setup, no_wandb: bool = False):
    # Overwrite args from command line if provided
    if no_wandb:
        setup.wandb.enable = False
    # if args.resume is not Nones:
    #     train_setup.resume_from_run

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

    train_step = SupervisedTrainingStep(
        synth["train"],
        model,
        criterion["train"],
        optimizer,
        setup.trainer_gradient_accumulation_steps,
        setup.enable_amp,
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
    brainnet.train.utilities.write_example_to_disk(
        trainer, evaluators, setup.results, event_handlers.write_input_image_with_affine
    )
    brainnet.train.utilities.write_example_to_disk(
        trainer, evaluators, setup.results, event_handlers.write_template
    )
    brainnet.train.utilities.load_checkpoint_from_setup(to_save, setup)

    print("Setup completed.", end="\n\n")

    print(f"Project             {setup.project:30s}")
    print(f"Contrast            {setup.contrast:30s}")
    print(f"Resolution          {setup.resolution:30s}")
    print(f"Run                 {setup.run:30s}")
    print(f"Load checkpoint     {setup.load_checkpoint:d}")
    print(f"Max epochs          {setup.max_epochs:d}")
    print(f"Output dir          {setup.results.out_dir}")
    print(f"Wandb enabled       {setup.wandb.enable}")
    print()

    return trainer, dataloader["train"]


def train(args):
    """

    python brainnet/train/alignment.py t1w_1mm --no-wandb

    args = brainnet.train.utilities.argparser_topofit(
        "brainnet/train/alignment.py synth_1mm --no-wandb".split()
    )

    """

    print(f"Using training specs: {args.specs}", end="\n\n")

    specs = importlib.import_module(f".{args.specs}", "brainnet.config.alignment")

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
    args = brainnet.train.utilities.argparser_topofit(sys.argv)
    train(args)
