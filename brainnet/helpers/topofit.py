from pathlib import Path

from ignite.engine import Engine
import nibabel as nib
import torch
import tqdm

import cortech

import brainsynth
from brainsynth.transforms.utils import recursive_function

import brainnet.networks
import brainnet.resources
from brainnet.config import TrainParameters
from brainnet.dict_utils import recursive_dict_sum
import brainnet.helpers.utils
from . import utils_bin
from .trega import PredictionStep as trega_PredictionStep

from brainnet.mesh.surface import load_deepsurfer_template
from brainnet import event_handlers, Surface
import brainnet.initializers

from brainnet.datasets import TopoFitDataset


# Define some recursive versions of functions
recursive_apply_affine = recursive_function(Surface.apply_affine)
recursive_float = recursive_function(torch.Tensor.float)
# recursive_Module_to = recursive_function(torch.nn.Module.to)
recursive_item = recursive_function(torch.Tensor.item)

DESCRIPTION = (
    "TopoFit is a network that predicts cortical surfaces (i.e., at the"
    "white-gray matter and gray matter-CSF interfaces)."
)


class Step:
    def __init__(
        self,
        preprocessor: None | brainsynth.Synthesizer,
        model,
        enable_amp: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        device = torch.device(device)
        self.preprocessor = preprocessor
        self.model = model
        self.model.to(device)
        self.enable_amp = enable_amp and not (device.type == "cpu")
        self.device = device
        self.set_prediction_topologies()

    def update_surface_template(self, template, data):
        """Insert vertices data from `data` into template and replace `data`
        with the template.
        """
        for typ, surfaces in data.items():
            for hemi in surfaces:
                template[typ][hemi].vertices = surfaces[hemi]
                data[typ][hemi] = template[typ][hemi]

    def set_prediction_topologies(self):
        self.topology = self.model.graph.out_topology

    def get_surfaces(self, vertices, vertex_data: dict | None = None):
        surfaces = {}
        for typ, surfs in vertices.items():
            surfaces[typ] = {}
            for hemi, v in surfs.items():
                surf = Surface(v, self.topology[hemi])
                if vertex_data is not None:
                    surf.vertex_data = vertex_data[typ][hemi]
                surfaces[typ][hemi] = surf
        return surfaces

    def prepare_batch(
        self, images, vox2ras, surfaces
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
        dict[str, dict[str, torch.Tensor]],
    ]:
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        if self.preprocessor is None:
            # assume synthesizer was applied when loading the data
            template = surfaces.pop("template")
            y_true = {k: surfaces[k] for k in ("white", "pial")}
            return images["image"], vox2ras["image"], template, y_true
        else:
            with torch.no_grad():
                out = self.preprocessor(images, vox2ras, surfaces)

            image = out.image
            vox2ras = out.affine
            template = out.surfaces["template"]
            y_true = {k: out.surfaces[k] for k in ("white", "pial")}

            return image, vox2ras, template, y_true

    def _amp_prediction(self, image: torch.Tensor, template: dict[str, torch.Tensor]):
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            features = self.model.unet(image)

        # with torch.autocast(self.device.type, torch.float32, enabled=self.enable_amp):
        features = recursive_float(features)
        out = self.model.graph(features, template)
        if self.model._swap_output:
            out = self.model.swap_output_levels(out)
        return out

    def postprocess(self, y_pred):
        return y_pred


class TrainingStep(Step):
    def __init__(
        self,
        preprocessor: brainsynth.Synthesizer,
        model,
        criterion: brainnet.Criterion,
        optimizer: torch.optim.Optimizer,
        enable_amp: bool = True,
        device: str | torch.device = "cpu",
        gradient_accumulation_steps: int = 1,
        freeze_body: bool = False,
    ) -> None:
        super().__init__(preprocessor, model, enable_amp, device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.freeze_body = freeze_body
        if self.freeze_body:
            self.model.unet.requires_grad_(False)

        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.enable_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

        self.y_true_reg = load_deepsurfer_template(
            self.model.graph.out_order, "sphere.reg"
        )
        self.y_true_reg.to(self.device)

    def _amp_compute_loss(self, y_pred, y_true):
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            self.criterion.prepare_for_surface_loss(y_pred, y_true)

            # recursive_Module_to(y_pred, torch.float32)
            # recursive_Module_to(y_true, torch.float32)

            # swap to {hemi: {surface: ...}} for loss calculation
            y_pred = self.model.swap_output_levels(y_pred)
            y_true = self.model.swap_output_levels(y_true)
            return self.criterion(y_pred, y_true)

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        image, vox2ras, template, y_true = self.prepare_batch(*batch)

        y_true = self.get_surfaces(y_true)
        y_true["registration"] = self.y_true_reg

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward

        y_pred = self._amp_prediction(image, template)
        y_pred = self.postprocess(y_pred)

        loss = self._amp_compute_loss(y_pred, y_true)
        loss = dict(raw=loss, weighted=self.criterion.apply_weights(loss))
        total_loss = recursive_dict_sum(loss["weighted"])
        total_loss /= self.gradient_accumulation_steps

        del y_true["registration"]  # no need to save this

        # exit if loss diverges
        if total_loss > 1e6 or torch.isnan(total_loss):
            raise RuntimeError(f"Loss diverged (loss = {total_loss}).\n{loss}")

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

        loss = recursive_item(loss)

        # these are stored in engine.state.output
        return loss, image, vox2ras, y_pred, y_true


class EvaluationStep(Step):
    def __init__(
        self,
        preprocessor: brainsynth.Synthesizer,
        model,
        criterion: brainnet.Criterion,
        enable_amp: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__(preprocessor, model, enable_amp, device)
        self.criterion = criterion

        self.y_true_reg = load_deepsurfer_template(
            self.model.graph.out_order, "sphere.reg"
        )
        self.y_true_reg.to(self.device)

    def _amp_compute_loss(self, y_pred, y_true):
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            self.criterion.prepare_for_surface_loss(y_pred, y_true)
            # recursive_Module_to(y_pred, torch.float32)
            # recursive_Module_to(y_true, torch.float32)

            # swap to {hemi: {surface: ...}} for loss calculation
            y_pred = self.model.swap_output_levels(y_pred)
            y_true = self.model.swap_output_levels(y_true)
            return self.criterion(y_pred, y_true)

    def __call__(self, engine, batch: tuple):
        self.model.eval()

        image, vox2ras, template, y_true = self.prepare_batch(*batch)
        y_true = self.get_surfaces(y_true)
        y_true["registration"] = self.y_true_reg

        with torch.inference_mode():
            y_pred = self._amp_prediction(image, template)
            y_pred = self.postprocess(y_pred)

            loss = self._amp_compute_loss(y_pred, y_true)
            loss = recursive_item(loss)
        del y_true["registration"]

        return loss, image, vox2ras, y_pred, y_true


class PredictionStep(Step):
    def __init__(self, trega_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trega_step = trega_step

    def prepare_batch(
        self,
        image: torch.Tensor,
        vox2ras: torch.Tensor,
        template: dict[str, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
    ]:
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        if self.preprocessor is not None:
            images = dict(image=image)
            affines = dict(image=vox2ras)
            surfaces = None if template is None else dict(template=template)
            with torch.no_grad():
                out = self.preprocessor(images, affines, surfaces)
            image = out.image
            vox2ras = out.affine
            template = out.surfaces["template"]
        # else assume preprocessor was applied when loading the data
        return image, vox2ras, template

    def postprocess(self, y_pred, vox2ras):
        # we don't want to transform the spherical registration
        sphere_reg = y_pred.pop("registration")
        _ = recursive_apply_affine(y_pred, vox2ras, inplace=True)
        y_pred["registration"] = sphere_reg
        return y_pred

    def __call__(self, engine, batch):
        self.model.eval()

        image, vox2ras, template = batch

        if template is None:
            image = image.to(self.device)
            vox2ras = vox2ras.to(self.device)
            y_pred = self.trega_step(None, (image, vox2ras))
            # to subject voxel space
            template = self.trega_step.apply_affine(
                torch.linalg.inv(vox2ras) @ y_pred["brain"],
                return_surface=False,
            )

        image, vox2ras, template = self.prepare_batch(image, vox2ras, template)

        with torch.inference_mode():
            y_pred = self._amp_prediction(image, template)
            y_pred = self.postprocess(y_pred, vox2ras)

        return y_pred

    @staticmethod
    def _preprocessor_from_config(contrast, resolution, device):
        config = brainnet.resources.load_pretrained_config(
            "topofit", contrast, resolution
        )
        config = brainsynth.config.PredictionConfig(
            "PredictionBuilder",
            config["preprocessor"]["out_size"],
            config["preprocessor"]["out_center_str"],
            device=device,
        )
        return brainsynth.Synthesizer(config)

    @classmethod
    def from_pretrained(
        cls,
        contrast,
        resolution,
        device: str | torch.device = "cpu",
        trega_kwargs: dict | None = None,
    ):
        kw = trega_kwargs or {}
        preprocessor = cls._preprocessor_from_config(contrast, resolution, device)
        model = brainnet.networks.TopoFit.from_pretrained(contrast, resolution, device)
        trega_step = trega_PredictionStep.from_pretrained(device=device, **kw)
        return cls(trega_step, preprocessor, model, device=device)


def write_example_prediction(y, out_dir):
    event_handlers.write_surfaces(y, out_dir)


def write_example_event(
    engine: Engine,
    evaluators: dict[str, Engine],
    config: brainnet.config.ResultsParameters,
):
    vol_info = event_handlers.FREESURFER_VOLUME_INFO

    for prefix, e in (dict(trainer=engine) | evaluators).items():
        prefix = f"epoch-{engine.state.epoch:05d}.{prefix}"
        match e._process_function:
            case TrainingStep():
                _, image, vox2ras, y_pred, y_true = e.state.output
            case EvaluationStep():
                _, image, vox2ras, y_pred, y_true = e.state.output
            # case PredictionStep():
            #     y_pred = e.state.output
            case _:
                raise ValueError(
                    f"Unknown _process_function type {type(e._process_function)}"
                )

        vol_info["volume"] = tuple(image.shape[-3:])

        # vox2ras is invalid
        if vox2ras is None:
            vox2ras = torch.eye(4)[None]
            vox2ras = vox2ras.broadcast_to(image.shape[0], *vox2ras.shape[1:])

        # write the volume that was used for prediction
        event_handlers.write_volume(
            image, vox2ras, config.examples_dir, prefix, label="image"
        )

        # write the predicted surfaces
        for tag, y in zip(("pred", "true"), (y_pred, y_true)):
            # for label, v in y.items():
            # if config.examples_keys is None or label in config.examples_keys:
            # if label == "surface":
            event_handlers.write_surfaces(
                # v["surface"],
                y,
                config.examples_dir,
                prefix,
                tag,
                vol_info=vol_info,
            )
            # else:
            # v = v[:, :5] if "dec:" in label else v
            # label = label.replace(":", "-")
            # event_handlers.write_volume(
            #     v, vox2ras, config.examples_dir, prefix, tag, label
            # )


def setup_model(setup):
    return setup.model


def create_trainer(
    setup: TrainParameters, no_wandb: bool = False
) -> tuple[Engine, torch.utils.data.DataLoader]:
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
        setup.device,
        setup.trainer_gradient_accumulation_steps,
        setup.UNET_FREEZE,
    )
    eval_step = EvaluationStep(
        synth["validation"],
        model,
        criterion["validation"],
        setup.enable_amp,
        setup.device,
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
    brainnet.helpers.utils.write_example_to_disk(
        trainer, evaluators, setup.results, write_example_event
    )
    brainnet.helpers.utils.load_checkpoint_from_setup(to_save, setup)

    print("Setup completed.", end="\n\n")
    print(setup)

    return trainer, dataloader["train"]


# def create_predictor(setup, datasets: list[str] | tuple | None = None):
#     model_helper = importlib.import_module(f"brainnet.helpers.{model}")
#     model = model_helper.setup_model(setup)

#     _valid_pred_contrasts = {"t1w", "t2w", "flair", "ct"}

#     # We need the affine as well
#     restrict_datasets_(setup, subset, datasets)
#     for v in setup.dataset[subset].dataset_kwargs.values():
#         found = [i for i in _valid_pred_contrasts if i in v["images"]]
#         n_found = len(found)
#         msg = f"{n_found} contrasts specified in dataset configuration ({found}) which is ambiguous for prediction. Need only one."
#         assert n_found == 1, msg
#         v["images"] = found

#     dataloaders = brainsynth.dataset.setup_dataloader(
#         setup.dataset[subset],
#         separate_datasets=True,
#         **setup.dataloader,
#         dataset_class=brainsynth.dataset.PredictionDataset,
#     )
#     if subset == "test":
#         out_size = setup.synthesizer["validation"].out_size
#         out_center_str = setup.synthesizer["validation"].out_center_str
#     elif subset == "testood":
#         out_size = setup.synthesizer["validation"].out_size
#         out_center_str = setup.synthesizer["validation"].out_center_str
#     else:
#         out_size = setup.synthesizer[subset].out_size
#         out_center_str = setup.synthesizer[subset].out_center_str
#     preprocessor = brainsynth.Synthesizer(
#         PredictionConfig("PredictionBuilder", out_size, out_center_str)
#     )

#     pred_step = model_helper.PredictionStep(
#         preprocessor, model, setup.enable_amp, setup.device
#     )
#     write_step = model_helper.write_example_prediction

#     to_load = dict(model=model)
#     brainnet.helpers.utils.load_checkpoint_from_setup(to_load, setup)

#     print("Setup completed.")
#     print(setup)

#     return pred_step, write_step, dataloaders


def predict(args):
    images = utils_bin._get_images(args.image)
    out_dirs = utils_bin._get_out_dirs(args.out_dir)
    transforms = utils_bin._get_transforms(args.transform)

    pred_step = PredictionStep.from_pretrained(
        args.contrast, args.resolution, args.device, trega_kwargs=dict(hemi=args.hemi)
    )
    in_order = pred_step.model.graph.active_topologies[0]
    dataset = TopoFitDataset(
        images,
        args.conform,
        transforms,
        args.mni_direction,
        args.mni_space,
        args.hemi,
        in_order,
    )

    for batch, out_dir in tqdm.tqdm(zip(dataset, out_dirs)):
        surfaces = pred_step(None, batch)

        if not (out_dir := Path(out_dir)).exists():
            out_dir.mkdir(parents=True)

        for name, hemispheres in surfaces.items():
            for h, surface in hemispheres.items():
                surface.to("cpu")
                v = surface.vertices.squeeze(0).numpy()
                f = surface.get_faces().numpy()
                cortech.Surface(v, f).save(out_dir / f"{h}.{name}")
                for k, curv in surface.vertex_data.items():
                    curv = curv.squeeze(0)
                    if curv.ndim > 1:
                        assert curv.ndim == 2
                        curv = torch.linalg.vector_norm(curv, axis=1)
                    curv = curv.cpu().numpy()
                    nib.freesurfer.write_morph_data(out_dir / f"{h}.{name}.{k}", curv)

        # if args.save_template:
        #     for h,vertices in template.items():
