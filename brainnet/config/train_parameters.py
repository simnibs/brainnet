from dataclasses import dataclass, InitVar
import importlib
from pathlib import Path

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
    load_checkpoint         : int                   = 0
    max_epochs              : int                   = 1
    device                  : str | torch.device    = "cuda"
    # Resume from this run. If None, resume from run defined by this setup
    resume_from_run         : InitVar[str | None]   = None
    epoch_length_train      : InitVar[int]          = 100
    epoch_length_val        : InitVar[int]          = 50
    out_dir                 : InitVar[Path | str]   = "/mnt/scratch/personal/jesperdn/results"
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
        out_dir,
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
        subject_subset_exclude,
        wandb_enable,
        wandb_run_id,
        wandb_run_tags,
        package,
    ):
        assert self.contrast in ("t1w", "synth")
        assert self.resolution in ("1mm", "random")

        data_dir = Path(data_dir)
        subjects_dir = Path(subjects_dir)
        out_dir = Path(out_dir)
        model_dir = model_dir or out_dir

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
            subject_dir=subjects_dir,
            exclude_subjects=subject_subset_exclude,
        )
        match self.contrast:
            case "synth":
                self.dataset_kwargs = dict(
                    train=kwargs_default
                    | dict(
                        subject_subset="train",
                        datasets=datasets,
                        images=["generation_labels_dist"],
                    ),
                    validation=kwargs_default
                    | dict(
                        subject_subset="validation",
                        datasets=datasets,
                        images=["generation_labels_dist", "t1w"],
                    ),
                )
            case "t1w":
                self.dataset_kwargs = dict(
                    train=kwargs_default
                    | dict(
                        subject_subset="train",
                        datasets=datasets,
                        images=["generation_labels_dist", "t1w"],
                    ),
                    validation=kwargs_default
                    | dict(
                        subject_subset="validation",
                        datasets=datasets,
                        images=["generation_labels_dist", "t1w"],
                    ),
                    exclude=kwargs_default
                    | dict(
                        subject_subset="exclude",
                        exclude_subjects=None,
                        datasets=datasets,
                        images=["generation_labels_dist", "t1w"],
                    ),
                )
            case "t2w":
                # HCP sub-059 excluded: T2w is just zeros!
                self.dataset_kwargs = dict(
                    train=kwargs_default
                    | dict(
                        subject_subset="train.t2",
                        datasets=["HCP", "OASIS3"],
                        images=["generation_labels_dist", "t2w"],
                    ),
                    validation=kwargs_default
                    | dict(
                        subject_subset="validation.t2",
                        datasets=["HCP", "OASIS3"],
                        images=["generation_labels_dist", "t2w"],
                    ),
                )
            case "flair":
                self.dataset_kwargs = dict(
                    train=kwargs_default
                    | dict(
                        subject_subset="train.flair",
                        datasets=["ADNI3", "AIBL"],
                        images=["generation_labels_dist", "flair"],
                    ),
                    validation=kwargs_default
                    | dict(
                        subject_subset="validation.flair",
                        datasets=["ADNI3", "AIBL"],
                        images=["generation_labels_dist", "flair"],
                    ),
                )
            case _:
                raise ValueError

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
        save_example_on = Events.EPOCH_COMPLETED(every=save_example_on_every)
        save_checkpoint_on = Events.EPOCH_COMPLETED(every=save_checkpoint_on_every)

        self.results = config.ResultsParameters(
            out_dir=out_dir / self.project / self.run,
            load_from_dir=model_dir / self.project / resume_from_run
            if resume_from_run is not None
            else None,
            save_example_on=save_example_on,
            save_checkpoint_on=save_checkpoint_on,
        )

        # =====================================================================
        # SYNTHESIZER
        # =====================================================================

        # =====================================================================
        # WANDB
        # =====================================================================

        self.wandb = config.WandbParameters(
            enable=wandb_enable,
            project=self.project,
            name=self.run,
            wandb_dir=out_dir / "wandb",
            log_on=self.evaluator_evaluate_on,
            run_id=wandb_run_id,
            tags=[self.contrast, self.resolution] + wandb_run_tags,
        )

        # Configuration parameters needs to be defined by the subclass
        self.dataloader = {}
        self.dataset = None
        self.model = None
        self.synthesizer = None

    def __str__(self):
        return "\n".join(
            [
                "Training parameters",
                "--------------------------------------",
                f"Project             {self.project:30s}",
                f"Contrast            {self.contrast:30s}",
                f"Resolution          {self.resolution:30s}",
                f"Run                 {self.run:30s}",
                f"Load checkpoint     {self.load_checkpoint:d}",
                f"Max epochs          {self.max_epochs:d}",
                f"Output dir          {self.results.out_dir}",
                f"Wandb enabled       {self.wandb.enable}",
            ]
        )
