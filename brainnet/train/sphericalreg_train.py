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

from brainnet.mesh.surface import TemplateSurfaces

from brainnet.sphere_utils import change_sphere_size


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

        self.radius_for_loss = 1.0 # 100.0
        self.radius_for_save = 100.0

        surfaces = ["sphere.reg"]

        self.y_true_template = {}
        self.y_pred_template = {}
        for h, top in self.model.topology.items():
            self.y_true_template[h] = {}
            self.y_pred_template[h] = {}
            for s in surfaces:
                self.y_true_template[h][s] = TemplateSurfaces(
                    torch.zeros(
                        (1, top.n_vertices, 3), device=self.device
                    ),
                    top,
                )
                self.y_pred_template[h][s] = TemplateSurfaces(
                    torch.zeros(
                        (1, top.n_vertices, 3), device=self.device
                    ),
                    top,
                )

    def as_surface(self, y_true, y_pred):
        """Insert vertices data from `data` into template and replace `data`
        with the template.
        """
        for h, surfaces in y_true.items():
            for s, v in surfaces.items():
                self.y_true_template[h][s].vertices = v
                y_true[h][s] = self.y_true_template[h][s]

        for h, surfaces in y_pred.items():
            for s, v in surfaces.items():
                self.y_pred_template[h][s].vertices = v
                y_pred[h][s] = self.y_pred_template[h][s]

    def prepare_batch(self, batch):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        _, y_true, init_verts = batch
        y_true = self.ensure_device(y_true)
        init_verts = self.ensure_device(init_verts)

        for k, v in y_true.items():
            for m, u in v.items():
                y_true[k][m] = u[:, : self.y_true_template[k][m].topology.n_vertices]
        for k, v in init_verts.items():
            init_verts[k] = v[:, : self.y_true_template[k][m].topology.n_vertices]

        return y_true, init_verts

    def prepare_for_loss_calculation(self, y_pred, y_true):
        """Ensure surfaces are stored like

            y["surface"]["lh"]["sphere.reg"]

        and that values are TemplateSurfaces.
        """
        # y_pred has keys lh, rh; reorder to y_pred[hemi][sphere.reg]

        y_pred = {h: {"sphere.reg": v} for h, v in y_pred.items()}

        self.as_surface(y_true, y_pred)

        # set radius
        for h, v in y_pred.items():
            for s in v:
                # y_pred[h][s].center_on_origin()
                change_sphere_size(y_pred[h][s], self.radius_for_loss)

        # center on origin; fix radius
        for h, v in y_true.items():
            for s in v:
                # y_true[h][s].center_on_origin()
                change_sphere_size(y_true[h][s], self.radius_for_loss)

        y_pred = dict(surface=y_pred)
        y_true = dict(surface=y_true)

        return y_pred, y_true

    def prepare_for_save(self, y_pred, y_true):
        if self.radius_for_save != self.radius_for_loss:
            self._prepare_for_save(y_pred)
            self._prepare_for_save(y_true)

    def _prepare_for_save(self, surf):
        for h, v in surf["surface"].items():
            for s in v:
                surf["surface"][h][s].center_on_origin()
                change_sphere_size(surf["surface"][h][s], self.radius_for_save)

    def compute_loss(self, y_pred, y_true):
        raw = self.criterion(y_pred, y_true)
        return dict(raw=raw, weighted=self.criterion.apply_weights(raw))


class SupervisedTrainingStep(SupervisedStep):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        gradient_accumulation_steps: int = 1,
        enable_amp: bool = False,
    ) -> None:
        super().__init__(model, criterion)
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.enable_amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        # sub,dsname=batch[-2:]
        # batch = batch[0]

        # no image
        y_true, init_verts = self.prepare_batch(batch)

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        # with torch.autograd.set_detect_anomaly(True):
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            y_pred = self.model(init_verts)
            n_batch = next(iter(y_pred.values())).shape[0]

            y_pred, y_true = self.prepare_for_loss_calculation(y_pred, y_true)
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

        self.prepare_for_save(y_pred, y_true)


        # these are stored in engine.state.output
        # the multiple of None is a hack...
        return loss, n_batch * [None], y_pred, y_true


class EvaluationStep(SupervisedStep):
    def __init__(self, model, criterion, enable_amp: bool = False):
        super().__init__(model, criterion)
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        self.model.eval()

        # sub,dsname=batch[-2:]
        # batch = batch[0]

        y_true, init_verts = self.prepare_batch(batch)

        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                y_pred = self.model(init_verts)
                n_batch = next(iter(y_pred.values())).shape[0]

                y_pred, y_true = self.prepare_for_loss_calculation(y_pred, y_true)
                loss = self.compute_loss(y_pred, y_true)

        # we don't need the weighted loss
        loss = recursive_itemize(loss["raw"])

        self.prepare_for_save(y_pred, y_true)

        return loss, n_batch * [None], y_pred, y_true


def train(args):
    """

    train_setup_file = "brainnet.config.braininflate.main"
    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")
    train_setup.wandb.enable = False

    args =  brainnet.train.utilities.parse_args("brainnet/train/sphericalreg_train.py brainnet.config.braininflate.main --load-checkpoint 40 --max-epochs 1600 --no-wandb".split())



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
    model = train_setup.model  # already initialize
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
    brainnet.train.utilities.write_example_to_disk(
        trainer, evaluators, train_setup.results, event_handlers.write_surfaces
    )
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
