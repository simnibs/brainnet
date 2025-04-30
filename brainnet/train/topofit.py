import copy
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
import brainnet.train.utilities
from brainnet import event_handlers
import brainnet.initializers
from brainnet.mesh.surface import Surface  # , load_deepsurfer_template
from brainnet.mesh.topology import DeepSurferTopology
from brainnet.modules.head import surface_modules

from brainnet.config.topofit import train_parameters


class SupervisedStep:
    def __init__(
        self,
        synthesizer: None | brainsynth.Synthesizer,
        model: brainnet.BrainNet,
        criterion: brainnet.Criterion,
        # subdivision: int,
    ) -> None:
        self.synthesizer = synthesizer
        self.model = model
        self.criterion = criterion
        self.device = self.model.device
        # self.set_templates(subdivision)
        self.set_prediction_topologies()

    def update_surface_template(self, template, data):
        """Insert vertices data from `data` into template and replace `data`
        with the template.
        """
        for h, surfaces in data.items():
            for s in surfaces:
                template[h][s].vertices = surfaces[s]
                data[h][s] = template[h][s]

    def set_prediction_topologies(self):
        module = [
            i for i in self.model.heads.values() if isinstance(i, surface_modules)
        ]
        if len(module) == 0:
            return {}

        assert len(module) == 1
        module = module[0]
        self.topology = copy.deepcopy(module.out_topology)

    def get_surfaces(self, vertices, vertex_data: dict | None = None):
        surfaces = {}
        for h, surfs in vertices.items():
            surfaces[h] = {}
            for s, v in surfs.items():
                surf = Surface(v, self.topology[h])
                if vertex_data is not None:
                    surf.vertex_data = vertex_data[h][s]
                surfaces[h][s] = surf
        return surfaces

    # def set_templates(self, subdivision: int):
    #     self.template_mni = load_deepsurfer_template(subdivision, self.device)
    #     self.template_sub = copy.deepcopy(self.template_mni)

    # def update_subject_template(self, affine):
    #     for h, s in self.template_mni.items():
    #         self.template_sub[h].vertices = apply_affine(affine, s.vertices)

    def prepare_batch(self, batch):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        if self.synthesizer is None:
            # assume synthesizer was applied when loading the data
            return batch
        else:
            images, affines, surfaces, init_verts = batch

            # Remove batch dim
            func = functools.partial(torch.squeeze, dim=0)
            images = recursively_apply_function(images, func)
            surfaces = recursively_apply_function(surfaces, func)
            init_verts = recursively_apply_function(init_verts, func)

            with torch.no_grad():
                y_true = self.synthesizer(
                    images, surfaces, init_verts, affines, unpack=False
                )

            # Add batch dim
            func = functools.partial(torch.unsqueeze, dim=0)
            y_true = recursively_apply_function(y_true, func)

            image = y_true.pop("image")
            init_verts = y_true.pop("initial_vertices")

            return image, y_true, init_verts

    def compute_loss(self, y_pred, y_true):
        self.criterion.prepare_for_surface_loss(y_pred["surface"], y_true["surface"])
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
        freeze_body: bool = False,
    ) -> None:
        super().__init__(synthesizer, model, criterion)
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        self.freeze_body = freeze_body
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.enable_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        image, y_true, template = self.prepare_batch(batch)

        # with torch.autocast(self.device.type, enabled=self.enable_amp):
        #     mni305_to_ras = self.model(image, vox2mri)["brain"]

        # template = apply_affine(mni305_to_ras, self.template)

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            # do loss calculations in float32
            # y_pred = self.model(image, template)
            # y_pred = recursively_apply_method(y_pred, "float")

            if self.freeze_body:
                with torch.no_grad():
                    features = self.model.body(image)
            else:
                features = self.model.body(image)
            features = recursively_apply_method(features, "float")
            y_pred = self.model.forward_heads(features, template)
            y_true["surface"] = self.get_surfaces(y_true["surface"])
            loss = self.compute_loss(y_pred, y_true)
            # print(loss["raw"]["white"]["neglogprob"])
            # print(loss["raw"]["pial"]["neglogprob"])
            # print()
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
        return loss, image, y_pred, y_true


class EvaluationStep(SupervisedStep):
    def __init__(self, synthesizer, model, criterion, enable_amp: bool = False):
        super().__init__(synthesizer, model, criterion)
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        self.model.eval()

        image, y_true, template = self.prepare_batch(batch)

        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                # y_pred = self.model(image, template)
                features = self.model.body(image)
                # we cast to float32 during training
                features = recursively_apply_method(features, "float")
                y_pred = self.model.forward_heads(features, template)
                y_true["surface"] = self.get_surfaces(y_true["surface"])
                loss = self.compute_loss(y_pred, y_true)

        # we don't need the weighted loss anymore
        loss = recursively_apply_method(loss["raw"], "item")

        return loss, image, y_pred, y_true


def setup_model(setup):
    return brainnet.initializers.init_model(setup.model)


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

    train_step = SupervisedTrainingStep(
        synth["train"],
        model,
        criterion["train"],
        optimizer,
        setup.trainer_gradient_accumulation_steps,
        setup.enable_amp,
        setup.UNET_FREEZE,
    )
    eval_step = EvaluationStep(
        synth["validation"],
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
    brainnet.train.utilities.add_metric_to_engine(trainer)
    brainnet.train.utilities.add_terminal_logger(trainer)

    # Add evaluations

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
    brainnet.train.utilities.write_example_to_disk(trainer, evaluators, setup.results)
    brainnet.train.utilities.load_checkpoint_from_setup(to_save, setup)

    print("Setup completed.", end="\n\n")
    print(setup)

    return trainer, dataloader["train"]


def train(args):
    """

    python brainnet/train/topofit.py synth_1mm --no-wandb

    args = brainnet.train.utilities.argparser_topofit(
        "brainnet/train/topofit.py t1w_1mm --no-wandb".split()
    )

    """

    print(f"Using training specs: {args.specs}", end="\n\n")

    specs = importlib.import_module(f".{args.specs}", "brainnet.config.topofit")

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
