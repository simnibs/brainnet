import copy
import functools
import importlib
import sys
import torch

from ignite.engine import Engine

import brainsynth

from brainnet.dict_utils import recursively_apply_function, recursively_apply_method, recursive_dict_sum
import brainnet.config
import brainnet.train.utilities
from brainnet import event_handlers
import brainnet.initializers
from brainnet.mesh.surface import Surface
from brainnet.modules.head import surface_modules

import importlib.util
f = "/home/jesperdn/repositories/BrainPass/scripts/demo_4jesper.py"
spec = importlib.util.spec_from_file_location("BrainPass", f)
brainpass = importlib.util.module_from_spec(spec)
sys.modules["BrainPass"] = brainpass
spec.loader.exec_module(brainpass)


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

        # Get empty TemplateSurfaces. We update the vertices at each iteration
        self.surface_template = dict(
            y_pred=self.get_placeholder_surface_templates(),
            y_true=self.get_placeholder_surface_templates(),
        )

    def get_placeholder_surface_templates(self):
        """Initialize placeholder objects for the predicted and target
        surfaces. The vertices are updated on each iteration.
        """
        surface_names = ("white", "pial")

        module = [i for i in self.model.heads.values() if isinstance(i, surface_modules)]

        if len(module) == 0:
            return None

        assert len(module) == 1
        module = module[0]

        # topology = module.get_prediction_topology()
        topology = module.out_topology
        topology = dict(lh=topology, rh=copy.deepcopy(topology))

        # only with andrew's template!
        # topology["rh"].reverse_face_orientation()

        return {
            h: {
                s: Surface(torch.zeros(t.n_vertices, 3, device=self.device), t)
                for s in surface_names
            }
            for h, t in topology.items()
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
        if self.synthesizer is None:
            # assume synthesizer was applied when loading the data
            return batch
        else:
            images, surfaces, init_verts = batch

            # Remove batch dim
            func = functools.partial(torch.squeeze, dim=0)
            images = recursively_apply_function(images, func)
            surfaces = recursively_apply_function(surfaces, func)
            init_verts = recursively_apply_function(init_verts, func)

            with torch.no_grad():
                y_true = self.synthesizer(images, surfaces, init_verts, unpack=False)

            # Add batch dim
            func = functools.partial(torch.unsqueeze, dim=0)
            y_true = recursively_apply_function(y_true, func)

            image = y_true.pop("image")
            init_verts = y_true.pop("initial_vertices")

            return image, y_true, init_verts

    def prepare_loss(self, y_pred, y_true):
        if (k := "surface") in y_pred:
            self.update_surface_template(self.surface_template["y_pred"], y_pred[k])
            self.update_surface_template(self.surface_template["y_true"], y_true[k])

            self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

    def compute_loss(self, y_pred, y_true):
        self.prepare_loss(y_pred, y_true)
        raw = self.criterion(y_pred, y_true)
        return dict(raw=raw, weighted=self.criterion.apply_weights(raw))


class SupervisedTrainingStep(SupervisedStep):
    def __init__(
        self, synthesizer, pretrained_model, model, criterion, optimizer,
        gradient_accumulation_steps: int = 1,
        enable_amp: bool = False
    ) -> None:
        super().__init__(synthesizer, model, criterion)
        self.optimizer = optimizer
        self.pretrained_model = pretrained_model
        self.enable_amp = enable_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.enable_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        image, y_true, init_verts = self.prepare_batch(batch)

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            # y_pred = self.model(image, init_verts)
            with torch.no_grad():
                features = self.pretrained_model(image)
            # features = self.model.body(image)

            features = dict(feats=features)
            features = {k:v.float() for k,v in features.items()}
            y_pred = self.model.forward_heads(features, init_verts)

            loss = self.compute_loss(y_pred, y_true)

            total_loss = recursive_dict_sum(loss["weighted"])
            total_loss /= self.gradient_accumulation_steps

        # wbefore = torch.clone(getattr(self.model.heads["surface"].pial_deform, "Convolution[out]").weight).detach()
        # wbefore1 = torch.clone(self.model.heads["surface"].white_deform["6"].transform[1].conv_self.weight).detach()

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

        # wafter = getattr(self.model.heads["surface"].pial_deform, "Convolution[out]").weight
        # assert not torch.allclose(wafter,wbefore)
        # wafter1 = self.model.heads["surface"].white_deform["6"].transform[1].conv_self.weight
        # assert not torch.allclose(wafter1,wbefore1)


        # these are stored in engine.state.output
        return loss, image, y_pred, y_true

class EvaluationStep(SupervisedStep):
    def __init__(self, synthesizer, pretrained_model, model, criterion, enable_amp: bool = False):
        super().__init__(synthesizer, model, criterion)
        self.pretrained_model = pretrained_model
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        self.model.eval()

        image, y_true, init_verts = self.prepare_batch(batch)

        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                features = self.pretrained_model(image)
            features = dict(feats=features)
            features = {k:v.float() for k,v in features.items()}
            y_pred = self.model.forward_heads(features, init_verts)
            loss = self.compute_loss(y_pred, y_true)

        # we don't need the weighted loss
        loss = recursively_apply_method(loss["raw"], "item")

        return loss, image, y_pred, y_true

def train(args):

    """

    args = brainnet.train.utilities.parse_args(
        "brainnet/train/train_peirong.py brainnet.config.topofit.brainpass.main --max-epochs 100 --no-wandb".split()
    )


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
    model = brainnet.initializers.init_model(train_setup.model)
    optimizer = brainnet.initializers.init_optimizer(train_setup.optimizer, model)
    synth = brainnet.initializers.init_synthesizer(train_setup.synthesizer)

    feat_model, processors, postprocessor, train_args, gen_args = brainpass.get_model()
    pretrained_model = functools.partial(
        brainpass.get_features,
        feat_model=feat_model,
        processors=processors,
        postprocessor=postprocessor,
        train_args=train_args,
        gen_args=gen_args,
        feat_only = True,
        win_size = None,
        )


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
    brainnet.train.utilities.write_example_to_disk(trainer, evaluators, train_setup.results)
    brainnet.train.utilities.load_checkpoint_from_setup(to_save, train_setup)

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
