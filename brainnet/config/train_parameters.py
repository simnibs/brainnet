from dataclasses import dataclass, InitVar
import importlib
import json
from pathlib import Path
from typing import Any

import torch
from ignite.engine import Events

from brainnet import config


class BaseObject:
    def __post_init__(self, *args, **kwargs):
        pass


@dataclass(kw_only=True)
class TrainParameters(BaseObject):
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """

    # fmt: off
    project                 : str
    contrast                : str
    resolution              : str
    run_suffix              : str                   = ""
    load_checkpoint         : int | str             = 0
    max_epochs              : int                   = 1
    device                  : str | torch.device    = "cuda"
    # Resume from this run. If None, resume from run defined by this setup
    resume_from_run         : InitVar[str | None]   = None
    epoch_length_train      : InitVar[int]          = 100
    epoch_length_val        : InitVar[int]          = 50
    results_dir             : Path | str            = "/mnt/scratch/personal/jesperdn/results"
    model_dir               : InitVar[Path | str | None]   = None
    # model_dir: Path = Path("/mnt/projects/CORTECH/nobackup/jesper/models")
    evaluate_on_every       : InitVar[int]          = 10
    save_example_on_every   : InitVar[int]          = 20
    save_checkpoint_on_every: InitVar[int]          = 20
    # Events defined in these files located in the same directory
    events_trainer          : InitVar[str]          = "events_trainer"
    events_evaluator        : InitVar[str]          = "events_evaluator"
    # Losses defined in this file located in the same directory
    losses                  : InitVar[str]          = "losses"

    # =========================================================================
    #   OPTIMIZER
    # =========================================================================

    optimizer_name          : InitVar[str]          = "AdamW"
    optimizer_kwargs        : InitVar[dict]         = dict(lr=1e-4)

    # =========================================================================
    #   MODEL
    # =========================================================================

    enable_amp: bool = True

    # =========================================================================
    #   DATASET
    # =========================================================================
    # COBRE and MCIC are used as test sets.
    # ISBI2015 results from FS are not great

    data_dir                : InitVar[Path | str]   = "/mnt/projects/CORTECH/nobackup/training_data/full"
    subject_dir             : InitVar[Path | str]   = "/mnt/projects/CORTECH/nobackup/training_data/subject_splits"
    datasets                : InitVar[list | tuple] = (
        "ABIDE",
        "ADHD200",
        "ADNI3",
        "AIBL",
        "Buckner40",
        "Chinese-HCP",
        # "COBRE",
        "HCP",
        # "ISBI2015",
        # "MCIC",
        "OASIS3",
    )
    images_train            : InitVar[list | tuple | None] = None
    images_validation       : InitVar[list | tuple | None] = None
    images_exclude          : InitVar[list | tuple | None] = None

    # {ds}.exclude.txt
    subject_subset_exclude: InitVar[str] = "exclude"

    # =========================================================================
    #   PREPROCESS
    # =========================================================================
    # Single hemisphere
    # target_surface_hemisphere: str = "lh"
    # out_size = [128, 224, 176]
    # out_center_str = "lh"

    # Full brain
    fov_out_size            : list | tuple          = (192, 224, 192)
    fov_out_center_str      : str                   = "image"

    builder_train: str | None = None
    builder_validation: str | None = None

    builder_train_kw: dict[str, Any] | None = None
    # do not use extracerebral augmentation (e.g., skullstripping) during
    # validation
    builder_validation_kw: dict[str, Any] | None = None

    selectable_images_train: list | tuple | None = None
    selectable_images_validation: list | tuple | None = None

    # =========================================================================
    #   WANDB
    # =========================================================================

    wandb_enable            : InitVar[bool]         = True
    wandb_run_id            : InitVar[str | None]   = None
    wandb_run_tags          : InitVar[list]         = []

    # For relative imports of losses and events
    package                 : InitVar[str]

    # fmt: on
    def __post_init__(
        self,
        resume_from_run,
        epoch_length_train,
        epoch_length_val,
        model_dir,
        evaluate_on_every,
        save_example_on_every,
        save_checkpoint_on_every,
        events_trainer,
        events_evaluator,
        losses,
        optimizer_name,
        optimizer_kwargs,
        data_dir,
        subjects_dir,
        datasets,
        images_train,
        images_validation,
        images_exclude,
        subject_subset_exclude,
        wandb_enable,
        wandb_run_id,
        wandb_run_tags,
        package,
    ):
        # assert self.contrast in ("t1w", "synth")
        # assert self.resolution in ("1mm", "random")

        data_dir = Path(data_dir)
        subjects_dir = Path(subjects_dir)
        self.results_dir = Path(self.results_dir)
        model_dir = model_dir or self.results_dir

        # Run name
        self.run: str = "-".join(
            [self.contrast, self.resolution, self.run_suffix]
        ).rstrip("-")

        resume_from_run = resume_from_run or self.run

        self.device = torch.device(self.device)

        # =====================================================================
        # CRITERION
        # =====================================================================

        losses = importlib.import_module(f".{losses}", package)

        self.criterion = dict(
            train=losses.train,
            validation=losses.validation,  # could/should be different...
        )

        # =====================================================================
        # TRAINING / EVALUATION PARAMETERS
        # =====================================================================

        train_events = importlib.import_module(f".{events_trainer}", package)
        eval_events = importlib.import_module(f".{events_evaluator}", package)

        self.trainer_epoch_length = epoch_length_train
        self.trainer_max_epochs = self.max_epochs
        self.trainer_gradient_accumulation_steps = 1
        self.trainer_events = train_events.events

        self.evaluator_epoch_length = epoch_length_val
        self.evaluator_evaluate_on = Events.EPOCH_COMPLETED(every=evaluate_on_every)
        self.evaluator_events = eval_events.events

        # =====================================================================
        # DATALOADER
        # =====================================================================

        # =====================================================================
        # DATASETS
        # =====================================================================

        kwargs_default = dict(
            root_dir=data_dir,
            datasets=datasets,
            subject_dir=subjects_dir,
            exclude_subjects=subject_subset_exclude,
        )
        images_exclude = images_exclude or images_validation
        match self.contrast:
            case "synth":
                self.dataset_kwargs = dict(
                    train=kwargs_default
                    | dict(
                        subject_subset="train",
                        images=images_train or ["generation_labels_dist"],
                    ),
                    validation=kwargs_default
                    | dict(
                        subject_subset="validation",
                        images=images_validation or ["generation_labels_dist", "t1w"],
                    ),
                )
            case "t1w":
                self.dataset_kwargs = dict(
                    train=kwargs_default
                    | dict(
                        subject_subset="train",
                        images=images_train or ["generation_labels_dist", "t1w"],
                    ),
                    validation=kwargs_default
                    | dict(
                        subject_subset="validation",
                        images=images_validation or ["generation_labels_dist", "t1w"],
                    ),
                    exclude=kwargs_default
                    | dict(
                        subject_subset="exclude",
                        exclude_subjects=None,
                        images=images_exclude or ["generation_labels_dist", "t1w"],
                    ),
                )
            case "t2w":
                # HCP sub-059 excluded: T2w is just zeros!
                self.dataset_kwargs = dict(
                    train=kwargs_default
                    | dict(
                        subject_subset="train.t2",
                        images=images_train or ["generation_labels_dist", "t2w"],
                    ),
                    validation=kwargs_default
                    | dict(
                        subject_subset="validation.t2",
                        images=images_validation or ["generation_labels_dist", "t2w"],
                    ),
                )
            case "flair":
                self.dataset_kwargs = dict(
                    train=kwargs_default
                    | dict(
                        subject_subset="train.flair",
                        images=images_train or ["generation_labels_dist", "flair"],
                    ),
                    validation=kwargs_default
                    | dict(
                        subject_subset="validation.flair",
                        images=images_validation or ["generation_labels_dist", "flair"],
                    ),
                )
            case _:
                raise ValueError

        if self.selectable_images_train is None:
            if self.contrast == "synth":
                self.selectable_images_train = None
            else:
                self.selectable_images_train = [self.contrast]
        elif isinstance(self.selectable_images_train, str):
            self.selectable_images_train = [self.selectable_images_train]

        if self.selectable_images_validation is None:
            if self.contrast == "synth":
                self.selectable_images_validation = ["t1w"]
            else:
                self.selectable_images_validation = [self.contrast]
        elif isinstance(self.selectable_images_validation, str):
            self.selectable_images_validation = [self.selectable_images_validation]

        # =====================================================================
        # MODEL
        # =====================================================================

        # =====================================================================
        # OPTIMIZER
        # =====================================================================

        self.optimizer = config.OptimizerParameters(optimizer_name, optimizer_kwargs)

        # =====================================================================
        # RESULTS
        # =====================================================================
        if save_example_on_every is None:
            save_example_on = None
        else:
            save_example_on = Events.EPOCH_COMPLETED(every=save_example_on_every)

        save_checkpoint_on = Events.EPOCH_COMPLETED(every=save_checkpoint_on_every)

        self.results = config.ResultsParameters(
            out_dir=self.results_dir / self.project / self.run,
            load_from_dir=model_dir / self.project / resume_from_run
            if resume_from_run is not None
            else None,
            save_example_on=save_example_on,
            save_checkpoint_on=save_checkpoint_on,
        )

        # =====================================================================
        # SYNTHESIZER
        # =====================================================================

        self.builder_train_kw = self.builder_train_kw or {}
        self.builder_validation_kw = self.builder_validation_kw or dict(
            intensity_transforms_kw=dict(extracerebral_augmentation=False)
        )

        # =====================================================================
        # WANDB
        # =====================================================================

        self.wandb = config.WandbParameters(
            enable=wandb_enable,
            project=self.project,
            name=self.run,
            wandb_dir=self.results_dir / "wandb",
            log_on=self.evaluator_evaluate_on,
            run_id=wandb_run_id,
            tags=[self.contrast, self.resolution] + wandb_run_tags,
        )

        # Configuration parameters needs to be defined by the subclass
        self.dataloader = {}
        self.dataset = None
        self.model = None
        self.synthesizer = None

        self.prediction_config = {}

    def dump_prediction_config(self, out_dir: Path | str, name: str | None = None):
        """Dump configuration information to a JSON file."""
        if name is None:
            name = f"{self.contrast}_{self.resolution}_config.json"
        filename = Path(out_dir) / name
        with open(filename, "w") as f:
            json.dump(self.prediction_config, f)

    def __str__(self):
        return "\n".join(
            [
                "Training parameters",
                "--------------------------------------",
                f"Project             {self.project:30s}",
                f"Contrast            {self.contrast:30s}",
                f"Resolution          {self.resolution:30s}",
                f"Run                 {self.run:30s}",
                f"Load checkpoint     {self.load_checkpoint}",
                f"Max epochs          {self.max_epochs:d}",
                f"Output dir          {self.results.out_dir}",
                f"Wandb enabled       {self.wandb.enable}",
            ]
        )
