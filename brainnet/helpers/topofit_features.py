import torch

from ignite.engine import Engine

import brainsynth

# from brainsynth.utilities import apply_affine
from brainsynth.transforms.utils import recursive_function

from brainnet.dict_utils import recursive_dict_sum
import brainnet.helpers.utils
from brainnet import event_handlers
import brainnet.initializers

from brainnet.config import TrainParameters

DESCRIPTION = "Train the image feature extractor of a TopoFit network"


# Define some recursive versions of functions
# recursive_Module_to = recursive_function(torch.nn.Module.to)
recursive_item = recursive_function(torch.Tensor.item)


def apply_brain_mask(y_pred, y_true, brain_dist_map, dist=10.0):
    """We only care about matching the features in the brain as these are the
    important ones.
    """
    mask = brain_dist_map < dist

    # a little inaccurate but does not matter so much as it is just
    # for weighting the losses
    subsamp = [2 ** (3 - int(k.split(":")[-1])) for k in y_true if "dec:" in k]
    mask = {k: mask[..., ::s, ::s, ::s].ravel() for k, s in zip(y_true, subsamp)}

    y_pred_masked = {
        k: v.reshape(*v.shape[:2], -1)[..., mask[k]] for k, v in y_pred.items()
    }
    y_true_masked = {
        k: v.reshape(*v.shape[:2], -1)[..., mask[k]] for k, v in y_true.items()
    }
    return y_pred_masked, y_true_masked


class Step:
    def __init__(
        self,
        preprocessor: None | brainsynth.Synthesizer,
        model,
        enable_amp: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        device = torch.device(device)
        if device.type == "cpu":
            assert not enable_amp, "Cannot use AMP with device type 'cpu'."

        self.preprocessor = preprocessor
        self.model = model
        self.model.to(device)
        self.enable_amp = enable_amp
        self.device = device

    def prepare_batch(
        self, images, vox2ras, surfaces
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        if self.preprocessor is None:
            # assume synthesizer was applied when loading the data
            return (
                images["image"],
                vox2ras["image"],
                images["t1w"],
                images["brain_dist_map"],
            )
        else:
            with torch.no_grad():
                out = self.preprocessor(images, vox2ras, surfaces)
            return (
                out.image,
                out.affine,
                out.images["t1w"],
                out.images["brain_dist_map"],
            )

    def _amp_prediction(self, image: torch.Tensor, model=None):
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            if model is None:
                features = self.model.unet(image)
            else:
                features = model.unet(image)
        return features

    def postprocess(self, y_pred):
        return y_pred


class TrainingStep(Step):
    def __init__(
        self,
        preprocessor: brainsynth.Synthesizer,
        model,
        pretrained_model,
        criterion: brainnet.Criterion,
        optimizer: torch.optim.Optimizer,
        enable_amp: bool = False,
        device: str | torch.device = "cpu",
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(preprocessor, model, enable_amp, device)
        self.pretrained_model = pretrained_model
        self.pretrained_model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.enable_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def _amp_compute_loss(self, y_pred, y_true):
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            # recursive_Module_to(y_pred, torch.float32)
            # recursive_Module_to(y_true, torch.float32)
            return self.criterion(y_pred, y_true)

    def __call__(self, engine, batch) -> tuple:
        self.model.train()
        self.pretrained_model.eval()

        image, vox2ras, t1w, brain_dist = self.prepare_batch(*batch)

        y_pred = self._amp_prediction(image)
        with torch.inference_mode():
            y_true = self._amp_prediction(t1w, self.pretrained_model)

        masked_y_pred, masked_y_true = apply_brain_mask(y_pred, y_true, brain_dist)

        loss = self._amp_compute_loss(masked_y_pred, masked_y_true)
        loss = dict(raw=loss, weighted=self.criterion.apply_weights(loss))
        total_loss = recursive_dict_sum(loss["weighted"])
        total_loss /= self.gradient_accumulation_steps

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
        pretrained_model,
        criterion: brainnet.Criterion,
        enable_amp: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__(preprocessor, model, enable_amp, device)
        self.pretrained_model = pretrained_model
        self.criterion = criterion

    def _amp_compute_loss(self, y_pred, y_true):
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            # recursive_Module_to(y_pred, torch.float32)
            # recursive_Module_to(y_true, torch.float32)
            return self.criterion(y_pred, y_true)

    def __call__(self, engine, batch: tuple):
        self.model.eval()
        self.pretrained_model.eval()

        image, vox2ras, t1w, brain_dist = self.prepare_batch(*batch)

        with torch.inference_mode():
            y_pred = self._amp_prediction(image)
            y_true = self._amp_prediction(t1w, self.pretrained_model)

        masked_y_pred, masked_y_true = apply_brain_mask(y_pred, y_true, brain_dist)

        loss = self._amp_compute_loss(masked_y_pred, masked_y_true)
        loss = recursive_item(loss)

        return loss, image, vox2ras, y_pred, y_true


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
        # event_handlers.write_volume(
        #     t1w, vox2ras, config.examples_dir, prefix, label="t1w"
        # )
        # write the predicted surfaces
        for tag, y in zip(("pred", "true"), (y_pred, y_true)):
            for label, v in y.items():
                if config.examples_keys is None or label in config.examples_keys:
                    v = v[:, :5] if "dec:" in label else v
                    label = label.replace(":", "-")
                    event_handlers.write_volume(
                        v, vox2ras, config.examples_dir, prefix, tag, label
                    )


def setup_model(setup):
    return setup.model


def setup_pretraind_model(setup):
    return setup.pretrained_model


def create_trainer(
    setup: TrainParameters, no_wandb: bool = False
) -> tuple[Engine, torch.utils.data.DataLoader]:
    # Overwrite args from command line if provided
    if no_wandb:
        setup.wandb.enable = False

    criterion = brainnet.initializers.init_criterion(setup.criterion)
    dataloader = brainnet.initializers.init_dataloader(setup.dataset, setup.dataloader)
    model = setup_model(setup)
    pretrained_model = setup_pretraind_model(setup)

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
        pretrained_model,
        criterion["train"],
        optimizer,
        setup.enable_amp,
        setup.device,
        setup.trainer_gradient_accumulation_steps,
    )
    eval_step = EvaluationStep(
        synth["validation"],
        model,
        pretrained_model,
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
    brainnet.helpers.utils.write_example_to_disk(
        trainer, evaluators, setup.results, write_example_event
    )

    brainnet.helpers.utils.load_checkpoint_from_setup(to_save, setup)
    brainnet.helpers.utils.load_checkpoint(
        dict(model=pretrained_model),
        setup.pretrained_checkpoint_filename,
        setup.device,
    )

    print("Setup completed.", end="\n\n")
    print(setup)

    return trainer, dataloader["train"]


# if __name__ == "__main__":
#     args = brainnet.helpers.utils.argparser_topofit(sys.argv)
#     train(args)
