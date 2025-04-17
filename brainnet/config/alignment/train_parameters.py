from dataclasses import dataclass, InitVar
import importlib
from pathlib import Path

import torch
from ignite.engine import Events

import brainsynth.config
from brainsynth.dataset import AlignmentDataset
from brainnet import config
from brainnet.modules.alignment import AffineCorticalAlignment


@dataclass
class TrainParameters:
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """

    # fmt: off
    project                 : str                   = "CorticalAlignment"
    contrast                : str                   = "t1w"  # synth, t1w, t2w, flair
    resolution              : str                   = "1mm" # 1mm, random
    run_suffix              : str                   = ""
    load_checkpoint         : int                   = 0
    max_epochs              : int                   = 1

    device                  : InitVar[str | torch.device] = "cuda"

    # Resume from this run. If None, resume from run defined by this setup
    resume_from_run         : InitVar[str | None]   = None

    epoch_length_train      : InitVar[int]          = 100
    epoch_length_val        : InitVar[int]          = 50

    out_dir                 : InitVar[str]          = "/mnt/scratch/personal/jesperdn/results"
    model_dir               : InitVar[str | None]   = None
    # model_dir: Path = Path("/mnt/projects/CORTECH/nobackup/jesper/models")

    evaluate_on_every       : InitVar[int]          = 10
    save_example_on_every   : InitVar[int]          = 20
    save_checkpoint_on_every: InitVar[int]          = 20

    # Events defined in these files located in the same directory
    events_trainer          : InitVar[str]  = "events_trainer"
    events_evaluator        : InitVar[str]  = "events_evaluator"
    # Losses defined in this file located in the same directory
    losses                  : InitVar[str]  = "losses"

    # =========================================================================
    #   OPTIMIZER
    # =========================================================================

    optimizer_name          : InitVar[str]          = "AdamW"
    optimizer_kwargs        : InitVar[dict]         = dict(lr=1e-4)

    # =========================================================================
    #   MODEL
    # =========================================================================

    # fmt: off
    UNET_ENCODER_CHANNELS   : InitVar[dict]         = {
        ("t1w", "1mm"):         [[16], [32], [64], [128], [256]],
        ("synth", "1mm"):       [[16], [32], [64], [128], [256]],
        ("synth", "random"):    [[16], [32], [64], [128], [256]],
    }
    UNET_DECODER_CHANNELS   : InitVar[dict]         = {
        ("t1w", "1mm"):         [[128], [64], [32], [16]],
        # ("t1w", "1mm"):         [[128], [64], [32], [32]],
        ("synth", "1mm"):       [[128], [96], [64], [64]],
        ("synth", "random"):    [[128], [96], [64], [64]],
    }
    # fmt: on
    UNET_RETURN_ENCODER_FEATURES: InitVar[list | None] = None
    UNET_RETURN_DECODER_FEATURES: InitVar[list | None] = [True, True, True, True]

    enable_amp: bool = True

    # =========================================================================
    #   DATASET
    # =========================================================================
    # COBRE and MCIC are used as test sets.
    # ISBI2015 results from FS are not great

    data_dir: InitVar[str] = "/mnt/projects/CORTECH/nobackup/training_data/full"
    subject_dir: InitVar[str] = (
        "/mnt/projects/CORTECH/nobackup/training_data/subject_splits"
    )
    datasets: InitVar[list] = [
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
    ]

    # =========================================================================
    #   PREPROCESS
    # =========================================================================
    # Single hemisphere
    # target_surface_hemisphere: str = "lh"
    # out_size = [128, 224, 176]
    # out_center_str = "lh"

    # Full brain
    fov_out_size: InitVar[list] = [192, 224, 192]
    fov_out_center_str: InitVar[str] = "image"

    # =========================================================================
    #   WANDB
    # =========================================================================

    wandb_run_id: InitVar[str | None] = None
    wandb_run_tags: InitVar[list] = []

    # fmt: on
    def __post_init__(
        self,
        device,
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
        UNET_ENCODER_CHANNELS,
        UNET_DECODER_CHANNELS,
        UNET_RETURN_ENCODER_FEATURES,
        UNET_RETURN_DECODER_FEATURES,
        data_dir,
        subjects_dir,
        datasets,
        fov_out_size,
        fov_out_center_str,
        wandb_run_id,
        wandb_run_tags,
    ):
        contrast = self.contrast
        resolution = self.resolution

        assert contrast in ("t1w", "synth")
        assert resolution in ("1mm", "random")

        data_dir = Path(data_dir)
        out_dir = Path(out_dir)
        model_dir = model_dir or out_dir

        # Run name
        self.run: str = "-".join([contrast, resolution, self.run_suffix]).rstrip("-")

        resume_from_run = resume_from_run or self.run

        wandb_run_tags += [contrast, resolution]

        device = torch.device(device)

        # =====================================================================
        # CRITERION
        # =====================================================================

        losses = importlib.import_module(f".{losses}", "brainnet.config.alignment")

        self.criterion = config.CriterionParameters(
            train=losses.cfg_loss,
            validation=losses.cfg_loss,  # could/should be different...
        )

        # =====================================================================
        # TRAINING / EVALUATION PARAMETERS
        # =====================================================================

        train_events = importlib.import_module(
            f".{events_trainer}", "brainnet.config.alignment"
        )
        eval_events = importlib.import_module(
            f".{events_evaluator}", "brainnet.config.alignment"
        )

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
        match contrast:
            case "synth":
                images_train = ["generation_labels_dist"]
                images_train_sel = None
                images_val = ["generation_labels_dist", "t1w"]
                images_val_sel = ["t1w"]
                subject_subset_train = "train"
                subject_subset_val = "validation"
            case "t1w":
                images_train = ["generation_labels_dist", "t1w"]
                images_train_sel = ["t1w"]
                images_val = ["generation_labels_dist", "t1w"]
                images_val_sel = ["t1w"]
                subject_subset_train = "train"
                subject_subset_val = "validation"
            case "t2w":
                images_train = ["generation_labels_dist", "t2w"]
                images_train_sel = ["t2w"]
                images_val = ["generation_labels_dist", "t2w"]
                images_val_sel = ["t2w"]
                # HCP sub-059 excluded: T2w is just zeros!
                subject_subset_train = "train.t2"
                subject_subset_val = "validation.t2"
                datasets = ["HCP", "OASIS3"]
            case "flair":
                images_train = ["generation_labels_dist", "flair"]
                images_train_sel = ["flair"]
                images_val = ["generation_labels_dist", "flair"]
                images_val_sel = ["flair"]
                subject_subset_train = "train.flair"
                subject_subset_val = "validation.flair"
                datasets = ["ADNI3", "AIBL"]
            case _:
                raise ValueError

        self.dataloader = dict(dataset_class=AlignmentDataset)

        # =====================================================================
        # DATASETS
        # =====================================================================
        # {ds}.exclude.txt
        subject_subset_exclude = "exclude"

        self.dataset = dict(
            train=brainsynth.config.DatasetConfig(
                root_dir=data_dir,
                subject_dir=subjects_dir,
                subject_subset=subject_subset_train,
                datasets=datasets,
                images=images_train,
                target_vertices=None,
                template_surface=None,
                exclude_subjects=subject_subset_exclude,
            ),
            validation=brainsynth.config.DatasetConfig(
                root_dir=data_dir,
                subject_dir=subjects_dir,
                subject_subset=subject_subset_val,
                datasets=datasets,
                images=images_val,
                target_vertices=None,
                template_surface=None,
                exclude_subjects=subject_subset_exclude,
            ),
        )

        # =====================================================================
        # MODEL
        # =====================================================================

        # unet = body.UNet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     encoder_channels=UNET_ENCODER_CHANNELS[contrast, resolution],
        #     decoder_channels=UNET_DECODER_CHANNELS[contrast, resolution],
        #     return_encoder_features=UNET_RETURN_ENCODER_FEATURES,
        #     return_decoder_features=UNET_RETURN_DECODER_FEATURES,
        #     # match the synth features to the T1w features
        #     encoder_post=None,
        #     decoder_post=UNET_DECODER_CHANNELS_POST[contrast, resolution],
        # )

        self.model = AffineCorticalAlignment(device=device)

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
        match contrast, resolution:
            case "t1w", "1mm":
                builder_train = "CropSelectIso"
                builder_validation = "CropSelectIso"
            case "synth", "1mm":
                builder_train = "CropSynthIso"
                builder_validation = "CropSelectIso"
            case "synth", "random":
                builder_train = "CropSynth"
                builder_validation = "CropSelect"

        self.synthesizer = config.SynthesizerParameters(
            train=brainsynth.config.SynthesizerConfig(
                builder=builder_train,
                out_size=fov_out_size,
                out_center_str=fov_out_center_str,
                # segmentation_labels = "brainseg"
                # photo_mode = False
                selectable_images=images_train_sel,
                device=device,
            ),
            validation=brainsynth.config.SynthesizerConfig(
                builder=builder_validation,
                out_size=fov_out_size,
                out_center_str=fov_out_center_str,
                # segmentation_labels = "brainseg"
                # photo_mode = False
                selectable_images=images_val_sel,  # ["t1w", "t2w", "flair"],
                device=device,
            ),
        )

        # =====================================================================
        # WANDB
        # =====================================================================

        self.wandb = config.WandbParameters(
            enable=self.enable_amp,
            project=self.project,
            name=self.run,
            wandb_dir=out_dir / "wandb",
            log_on=self.evaluator_evaluate_on,
            run_id=wandb_run_id,
            tags=wandb_run_tags,
        )
