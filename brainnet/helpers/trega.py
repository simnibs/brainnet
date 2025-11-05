import numpy as np
import torch

from ignite.engine import Engine
import tqdm

import brainsynth
from brainsynth.transforms import EnsureDevice
from brainsynth.transforms.utils import recursive_function

import brainnet.resources
from brainnet.config.trega.train_parameters import TrainParameters
from brainnet.dict_utils import recursive_dict_sum
import brainnet.helpers.utils
from brainnet import event_handlers
import brainnet.initializers
from brainnet.mesh.surface import load_deepsurfer_template, Surface
from . import utils_bin
from brainnet.datasets import ImageDataset, MNI305_to_MNI152


"""The following classes/functions need to be defined as they are imported
elsewhere.

Step
    Baseclass for the step.
TrainingStep
    Class defining actions in a training step, e.g., predict, loss and grad
    calculation, and step.
EvaluationStep
    Class defining actions in an evaluation step, e.g., predict and loss
    calculation.
PredictionStep


write_example_prediction
    Function which writes a prediction to disk.
write_example_event
    Event handler attachable to an Engine.

setup_model
    Function that returns a model from an instance of TrainingParameters
    (defining the model).

create_trainer
    Function that returns a trainer Engine and a dataloader.
"""

DESCRIPTION = (
    "TREGA (Template REGistration Affine) is a network that predicts an affine "
    "transformation between MNI305 and subject space."
)

# Define some recursive versions of functions
recursive_item = recursive_function(torch.Tensor.item)


class Step:
    def __init__(
        self,
        preprocessor: brainsynth.Synthesizer | None,
        model: torch.nn.Module,
        enable_amp: bool = True,
        device: str | torch.device = "cpu",
        template_resolution: int = 0,
        hemi: str = "both",
    ) -> None:
        assert hemi in {"both", "lh", "rh"}
        hemis = ["lh", "rh"] if hemi == "both" else [hemi]
        device = torch.device(device)
        self.device = device
        self.enable_amp = enable_amp and not (device.type == "cpu")
        self.preprocessor = preprocessor
        self.model = model
        self.model.to(device)
        self.ensure_device = EnsureDevice(device)

        self.template = load_deepsurfer_template(template_resolution, "white", hemis)
        self.template.to(device)
        self.topologies = {h: s.topology for h, s in self.template.items()}

    def apply_affine(self, affine: torch.Tensor, *args, **kwargs):
        """Apply an affine to the template."""
        return {
            k: v.apply_affine(affine, *args, **kwargs) for k, v in self.template.items()
        }

    def vertex_dict_to_surface_dict(self, templates):
        return {
            (hemi, affine): Surface(v, self.topologies[hemi])
            for (hemi, affine), v in templates.items()
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
            with torch.no_grad():
                out = self.preprocessor(images, vox2ras)
            return out.image, out.affine, y_true

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
        enable_amp: bool = True,
        device: str | torch.device = "cpu",
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(preprocessor, model, enable_amp, device)
        self.criterion = criterion
        self.optimizer = optimizer
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

        loss = recursive_item(loss)

        y_pred = self.vertex_dict_to_surface_dict(y_pred)
        y_true = self.vertex_dict_to_surface_dict(y_true)

        # these are stored in engine.state.output
        return loss, image, vox2ras, y_pred, y_true


class EvaluationStep(Step):
    def __init__(
        self,
        preprocessor,
        model,
        criterion,
        enable_amp: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__(preprocessor, model, enable_amp, device)
        self.criterion = criterion

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

        loss = recursive_item(loss)

        y_pred = self.vertex_dict_to_surface_dict(y_pred)
        y_true = self.vertex_dict_to_surface_dict(y_true)

        return loss, image, vox2ras, y_pred, y_true


class PredictionStep(Step):
    def __init__(self, preprocessor, model, *args, **kwargs):
        super().__init__(preprocessor, model, *args, **kwargs)

    def prepare_batch(
        self,
        image: torch.Tensor,
        vox2ras: torch.Tensor,
    ):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.

        The preprocessor needs to return the image and the associated
        voxel-to-ras transform.

        """
        if self.preprocessor is not None:
            with torch.no_grad():
                out = self.preprocessor(dict(image=image), dict(image=vox2ras))
            image = out.image
            vox2ras = out.affine
        # else assume preprocessor was applied when loading the data
        return image, vox2ras

    def postprocess(self, y_pred):
        return self.apply_affine(y_pred["brain"])

    def __call__(self, engine, batch):
        self.model.eval()

        image, vox2ras = self.prepare_batch(*batch)

        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                y_pred = self.model(image, vox2ras)

        return y_pred

    @staticmethod
    def _preprocessor_from_config(contrast, resolution, device):
        config = brainnet.resources.load_pretrained_config(
            "trega", contrast, resolution
        )
        config = brainsynth.config.PredictionConfig(
            "PredictionBuilder", **config["preprocessor"], device=device
        )
        return brainsynth.Synthesizer(config)

    @classmethod
    def from_pretrained(
        cls,
        contrast="synth",
        resolution="random",
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        device = torch.device(device)
        preprocessor = cls._preprocessor_from_config(contrast, resolution, device)
        model = brainnet.networks.TemplateRegAffine.from_pretrained(
            contrast, resolution, device
        )
        return cls(preprocessor, model, device=device, **kwargs)


def write_example_prediction(y, out_dir):
    event_handlers.write_surfaces(y, out_dir, surf="template")


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
    )
    eval_step = EvaluationStep(
        synth["validation"],
        model,
        criterion["validation"],
        setup.enable_amp,
        setup.device,
    )
    trainer = Engine(train_step)

    # The order in which the events are added to the engine is important!

    # Aggregate average loss over epoch
    brainnet.helpers.utils.add_metric_to_engine(trainer)
    brainnet.helpers.utils.add_terminal_logger(trainer)

    # Add evaluations
    evaluators = dict(
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
    # brainnet.helpers.utils.write_example_to_disk(
    #     trainer, evaluators, setup.results, event_handlers.write_input_image_with_affine
    # )
    brainnet.helpers.utils.write_example_to_disk(
        trainer, evaluators, setup.results, write_example_event
    )
    brainnet.helpers.utils.load_checkpoint_from_setup(to_save, setup)

    print("Setup completed.", end="\n\n")
    print(setup)

    return trainer, dataloader["train"]


def predict(args):
    images = utils_bin.get_images(args.image)
    out_dirs = utils_bin.get_out_dirs(args.out_dir)

    pred_step = PredictionStep.from_pretrained(device=args.device)
    dataset = ImageDataset(images, args.conform)

    for batch, out_dir in tqdm.tqdm(zip(dataset, out_dirs)):
        # batch = (image, vox2ras)
        y_pred = pred_step(None, batch)

        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        if args.all:
            for k, v in y_pred.items():
                v = v.squeeze(0).cpu().numpy()
                v = MNI305_to_MNI152 @ v if args.mni_space == "mni152" else v
                np.savetxt(out_dir / f"{args.mni_space}-to-ras.{k}.txt", v)
        else:
            v = y_pred["brain"].squeeze(0).cpu().numpy()
            v = MNI305_to_MNI152 @ v if args.mni_space == "mni152" else v
            np.savetxt(out_dir / f"{args.mni_space}-to-ras.txt", v)
