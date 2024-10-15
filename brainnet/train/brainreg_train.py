import copy
import importlib
from pathlib import Path
import sys
import torch

from ignite.engine import Engine

import brainsynth

import brainnet.config
import brainnet.train.utilities

from brainnet import event_handlers
import brainnet.initializers
from brainnet.mesh.surface import TemplateSurfaces
from brainnet.utilities import recursive_dict_sum, recursive_itemize

from brainnet.mesh import topology

from brainsynth.transforms import (
    EnsureDevice,
    IntensityNormalization,
)  # , OneHotEncoding


class SupervisedStep:
    def __init__(
        self,
        synthesizer: None | brainsynth.Synthesizer,
        model: brainnet.BrainReg,
        criterion: brainnet.Criterion,
        surface_resolution: int,
    ) -> None:
        self.synthesizer = synthesizer
        self.model = model
        self.criterion = criterion
        self.device = self.model.device

        self.intensity_normalization = IntensityNormalization(device=self.device)
        self.ensure_device = EnsureDevice(self.device)
        # self.onehot_enc = OneHotEncoding(57) # 0-56 in brainseg_with_extracerebral

        # Get empty TemplateSurfaces. We update the vertices at each iteration
        if surface_resolution is not None:
            self.surface_template = dict(
                y_pred=self.get_placeholder_surface_templates(surface_resolution),
                y_true=self.get_placeholder_surface_templates(surface_resolution),
            )

    def get_placeholder_surface_templates(self, surface_resolution):
        """Initialize placeholder objects for the predicted and target
        surfaces. The vertices are updated on each iteration.
        """
        surface_names = ("white", "pial")

        top = topology.get_recursively_subdivided_topology(
            surface_resolution, topology.initial_faces.to(self.device)
        )[-1]
        top = dict(lh=top, rh=copy.deepcopy(top))
        top["rh"].reverse_face_orientation()

        return {
            h: {
                s: TemplateSurfaces(torch.zeros(t.n_vertices, 3, device=self.device), t)
                for s in surface_names
            }
            for h, t in top.items()
        }

    def update_surface_template(self, template, data):
        """Insert vertices data from `data` into template and replace `data`
        with the template.
        """
        for h, surfaces in data.items():
            for s in surfaces:
                template[h][s].vertices = surfaces[s]
                data[h][s] = template[h][s]

    def prepare_batch(self, batch):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        # We don't need the initial vertices
        images, surfaces, _ = batch

        images = self.ensure_device(images)
        surfaces = self.ensure_device(surfaces)

        assert (
            len(next(iter(images))) % 2 == 0
        ), "Even number of examples in a batch required."

        # We abuse the batching done by the dataloader and reshape like
        #
        #   batch = (N,1,W,H,D) -> (N/2,2,W,H,D)
        #
        # such that we can use consecutive subjects are registration pairs.
        # for k, v in images.items():
        #     size = v.size()
        #     images[k] = v.reshape(size[0] // 2, 2, *size[2:])

        # for k, v in surfaces.items():
        #     for kk, vv in v.items():
        #         size = vv.size()
        #         surfaces[k][kk] = vv[None].reshape(size[0] // 2, 2, *size[2:])

        if self.synthesizer is None:
            # assume synthesizer was applied when loading the data

            for k, v in images.items():
                if v.is_floating_point():
                    images[k] = self.intensity_normalization(v)
                # elif k == "brainseg_with_extracerebral":
                #     images[k] = torch.stack([self.onehot_enc(vv) for vv in v])
            if len(surfaces) > 0:
                images["surface"] = surfaces
            return images["t1w_areg_mni"], images
        else:
            raise NotImplementedError("Synth with BrainReg is not yet implemented!")

            images, surfaces, init_verts = batch

            # REMOVE BATCH DIMENSION
            for k, v in images.items():
                images[k] = v.squeeze(0)
            for k, v in surfaces.items():
                for kk, vv in v.items():
                    surfaces[k][kk] = vv.squeeze(0)
            for k, v in init_verts.items():
                init_verts[k] = v.squeeze(0)

            with torch.no_grad():
                y_true = self.synthesizer(images, surfaces, init_verts, unpack=False)

            image = y_true.pop("image")
            init_verts = y_true.pop("initial_vertices")

            # ADD BATCH DIMENSION
            image = image.unsqueeze(0)
            for k, v in y_true.items():
                if k == "surface":
                    for k2, v2 in v.items():
                        for k3, v3 in v2.items():
                            y_true[k][k2][k3] = v3.unsqueeze(0)
                else:
                    y_true[k] = v.unsqueeze(0)
            for k, v in init_verts.items():
                init_verts[k] = v.unsqueeze(0)

            return image, y_true, init_verts

    def prepare_loss(self, y_pred, y_true):
        if (k := "surface") in y_pred:
            self.update_surface_template(self.surface_template["y_pred"], y_pred[k])
            self.update_surface_template(self.surface_template["y_true"], y_true[k])

            self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

    def compute_loss(self, y_pred, y_true, compute_weighted_loss=True):
        self.prepare_loss(y_pred, y_true)
        raw = self.criterion(y_pred, y_true)
        loss = dict(raw=raw)
        if compute_weighted_loss:
            loss["weighted"] = self.criterion.apply_weights(raw)
        return loss

    def predict(self, images: torch.Tensor, y_true: dict):

        # batch (N, 1, ...) was reshaped as (N/2, 2, ...) thus
        # y_pred = (N/2, 3, ...)
        y_pred = dict(svf=self.model(images))

        # deform 0 to 1

        # we swap 0 and 1 when returning the predicted image(s) and surfaces.
        # This way, they can be directly compared with y_true

        # y_true has [sub-01, sub-02] in channel dim but y_pred is returned
        # as [deformed(sub-01), deformed(sub-02)] in channel dim so we need
        # to swap either y_pred or y_true when we calculate the losses
        # (such that deformed(sub-01) aligns with sub-02 etc.)

        deform_fwd, deform_bwd = self.model.integrate_svf(y_pred["svf"])

        # stack predictions inversely:
        # align the prediction of 0 to image 1 and prediction of 1 to image 1
        for k0, v0 in y_true.items():
            if k0 == "surface":
                y_pred[k0] = {}
                for k1, v1 in v0.items():  # hemi
                    y_pred[k0][k1] = {}
                    for k2, v2 in v1.items():  # surfaces
                        s0 = v2[0::2]  # (N, V, 3) -> (N/2, V, 3)
                        s1 = v2[1::2]

                        s0 = self.model.deform_surface(s0, deform_bwd)
                        s1 = self.model.deform_surface(s1, deform_fwd)

                        y_pred[k0][k1][k2] = torch.zeros_like(v2)
                        y_pred[k0][k1][k2][0::2] = s1
                        y_pred[k0][k1][k2][1::2] = s0

            else:
                i0 = v0[0::2]  # (N/2, 2, ...) -> (N/2, 1, ...)
                i1 = v0[1::2]

                i0 = self.model.deform_image(i0, deform_fwd)
                i1 = self.model.deform_image(i1, deform_bwd)

                y_pred[k0] = torch.zeros_like(v0)
                y_pred[k0][0::2] = i1
                y_pred[k0][1::2] = i0

        return y_pred

class SupervisedTrainingStep(SupervisedStep):
    def __init__(
        self,
        synthesizer,
        model,
        criterion,
        optimizer,
        surface_resolution: int,
        gradient_accumulation_steps: int = 1,
        enable_amp: bool = False,
    ) -> None:
        super().__init__(synthesizer, model, criterion, surface_resolution)
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_amp = enable_amp
        if self.enable_amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def __call__(self, engine, batch) -> tuple:
        images, y_true = self.prepare_batch(batch)

        self.model.train()

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as was used during forward
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            y_pred = self.predict(images, y_true)
            loss = self.compute_loss(y_pred, y_true)

        total_loss = recursive_dict_sum(loss["weighted"])
        total_loss /= self.gradient_accumulation_steps
        loss_item = recursive_itemize(loss)

        # exit if loss diverges
        if total_loss > 1e6 or torch.isnan(total_loss):
            raise RuntimeError(f"Loss diverged:\n\n{loss_item})")

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

        # these are stored in engine.state.output
        return loss_item, images, y_pred, y_true


class EvaluationStep(SupervisedStep):
    def __init__(
        self,
        synthesizer,
        model,
        criterion,
        surface_resolution,
        enable_amp: bool = False,
    ):
        super().__init__(synthesizer, model, criterion, surface_resolution)
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        images, y_true = self.prepare_batch(batch)

        self.model.eval()
        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                y_pred = self.predict(images, y_true)
                loss = self.compute_loss(y_pred, y_true, compute_weighted_loss=False)


        loss = recursive_itemize(loss["raw"])

        return loss, images, y_pred, y_true


def train(args):
    # args.config

    train_setup_file = args.config  # "brainnet.config.surface_model.main"

    print("Setting up training...")

    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")

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
    model = brainnet.initializers.init_model(train_setup.model)
    optimizer = brainnet.initializers.init_optimizer(train_setup.optimizer, model)
    synth = brainnet.initializers.init_synthesizer(train_setup.synthesizer)

    # =============================================================================
    # TRAINING
    # =============================================================================

    surface_resolution = {
        k: next(iter(v.dataset_kwargs.values()))["target_surface_resolution"]
        for k, v in vars(train_setup.dataset).items()
    }

    train_step = SupervisedTrainingStep(
        synth["train"],
        model,
        criterion["train"],
        optimizer,
        surface_resolution["train"],
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
        # train = add_evaluation_event(
        #     criterion=criterion["validation"],  # NOTE here we use the validation criterion!
        #     synth=synth["train"],
        #     dataloader=dataloader["train"],
        #     logger=event_handlers.MetricLogger(key="loss", name="train"),
        #     **kwargs,
        # ),
        validation=brainnet.train.utilities.add_evaluation_event(
            EvaluationStep(
                synth["validation"],
                model,
                criterion["validation"],
                surface_resolution["validation"],
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

    # to_save = dict(model=model, optimizer=optimizer, engine=trainer)
    # load_checkpoint(to_save, train_setup)

    # Include this in the checkpoint
    to_save = dict(
        model=model,
        optimizer=optimizer,
        engine=trainer,
        **{f"criterion[{k}]": v for k, v in criterion.items()},
    )

    brainnet.train.utilities.add_model_checkpoint(trainer, to_save, train_setup.results)
    brainnet.train.utilities.write_example_to_disk(
        trainer, evaluators, train_setup.results
    )

    brainnet.train.utilities.load_checkpoint(to_save, train_setup)

    print("Setup completed. Starting training at epoch ...")

    sep_line = 79 * "="

    print(sep_line)
    print(f"Config file     {train_setup_file}")
    print(f"Project         {train_setup.project:30s}")
    print(f"Run             {train_setup.run:30s}")
    print(f"Output dir      {train_setup.results.out_dir}")
    print(f"Wandb enabled   {train_setup.wandb.enable}")
    print(sep_line)

    # Start the training
    trainer.run(
        dataloader["train"],
        epoch_length=train_setup.train_params.epoch_length_train,
        max_epochs=train_setup.train_params.max_epochs,
    )


if __name__ == "__main__":
    args = brainnet.train.utilities.parse_args(sys.argv)
    train(args)
