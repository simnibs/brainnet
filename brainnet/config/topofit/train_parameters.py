
from dataclasses import dataclass, InitVar
import importlib
from pathlib import Path

import torch
from ignite.engine import Events

import brainsynth.config
from brainnet import config
from brainnet.modules import body, head

@dataclass
class TrainParameters:
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """
    # fmt: off
    project                 : str                   = "TopoFit"
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
    optimizer_kwargs        : InitVar[dict]         = dict(lr=5e-5)

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

    UNET_FREEZE             : bool                  = False

    TOPOFIT_ORDER_IN        : InitVar[int]          = 0
    TOPOFIT_ORDER_OUT       : InitVar[int]          = 6
    TOPOFIT_ORDER_MAX       : InitVar[int]          = 6
    TOPOFIT_WHITE_MATTER_CHANNELS: InitVar[dict]    = dict(
        encoder=[96, 96, 96, 96],
        decoder=[96, 96, 96],
    )
    TOPOFIT_GRAY_MATTER_CHANNELS: InitVar[list]     = [32]
    TOPOFIT_GRAY_MATTER_MODULE: InitVar[str]        = "LinearDeformationBlock"

    enable_amp              : bool                  = True

    # =========================================================================
    #   DATASET
    # =========================================================================
    # COBRE and MCIC are used as test sets.
    # ISBI2015 results from FS are not great

    data_dir                : InitVar[str]          = "/mnt/projects/CORTECH/nobackup/training_data/full"
    subject_dir             : InitVar[str]          = "/mnt/projects/CORTECH/nobackup/training_data/subject_splits"
    datasets                : InitVar[list]         = [
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
    fov_out_size            : InitVar[list]         = [176, 208, 176]
    fov_out_center_str      : InitVar[str]          = "brain"

    # =========================================================================
    #   PRETRAINED
    # =========================================================================

    load_body_from_checkpoint : InitVar[str | None] = "/mnt/scratch/personal/jesperdn/results/TopoFit-UNet/synth_1mm/checkpoint/state_checkpoint_00400.pt"
    load_head_from_checkpoint : InitVar[str | None] = None # "/mnt/scratch/personal/jesperdn/results/TopoFit/t1w_1mm-taubin_16/checkpoint/state_checkpoint_00600.pt"

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
        save_checkpoint_on_every,
        save_example_on_every,
        events_trainer,
        events_evaluator,
        losses,
        optimizer_name,
        optimizer_kwargs,
        UNET_ENCODER_CHANNELS,
        UNET_DECODER_CHANNELS,
        UNET_RETURN_ENCODER_FEATURES,
        UNET_RETURN_DECODER_FEATURES,
        TOPOFIT_ORDER_IN,
        TOPOFIT_ORDER_OUT,
        TOPOFIT_ORDER_MAX,
        TOPOFIT_WHITE_MATTER_CHANNELS,
        TOPOFIT_GRAY_MATTER_CHANNELS,
        TOPOFIT_GRAY_MATTER_MODULE,
        data_dir,
        subjects_dir,
        datasets,
        fov_out_size,
        fov_out_center_str,
        load_body_from_checkpoint,
        load_head_from_checkpoint,
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

        match resolution:
            case "1mm":
                builder_res = "Iso"
            case "random":
                builder_res = ""
            case _:
                raise ValueError


        # =====================================================================
        # CRITERION
        # =====================================================================

        losses = importlib.import_module(f".{losses}", "brainnet.config.topofit")


        self.criterion = config.CriterionParameters(
            train=losses.cfg_loss,
            validation=losses.cfg_loss,  # could/should be different...
        )

        # =====================================================================
        # TRAINING / EVALUATION PARAMETERS
        # =====================================================================


        train_events = importlib.import_module(f".{events_trainer}", "brainnet.config.topofit")
        eval_events = importlib.import_module(f".{events_evaluator}", "brainnet.config.topofit")

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

        random_skullstrip: bool = True

        builder_contrast = "Synth" if contrast == "synth" else "Select"
        if builder_contrast == "Synth" or random_skullstrip:
            # synth has skullstrip anyway
            builder_train = f"Only{builder_contrast}{builder_res}"
        else:
            builder_train = f"Only{builder_contrast}NoSkullStrip{builder_res}"
        builder_validation = f"OnlySelectNoSkullStrip{builder_res}"

        self.dataloader = config.DataloaderParameters()

        # =====================================================================
        # DATASETS
        # =====================================================================
        # {ds}.exclude.txt
        subject_subset_exclude = "exclude"

        template_surface = dict(resolution=TOPOFIT_ORDER_IN, name="template")
        target_vertices = dict(resolution=TOPOFIT_ORDER_OUT, name="target")

        self.dataset = config.DatasetParameters(
            train=brainsynth.config.DatasetConfig(
                root_dir=data_dir,
                subject_dir=subjects_dir,
                subject_subset=subject_subset_train,
                datasets=datasets,
                images=images_train,
                target_vertices=target_vertices,
                template_surface=template_surface,
                exclude_subjects=subject_subset_exclude,
            ),
            validation=brainsynth.config.DatasetConfig(
                root_dir=data_dir,
                subject_dir=subjects_dir,
                subject_subset=subject_subset_val,
                datasets=datasets,
                images=images_val,
                target_vertices=target_vertices,
                template_surface=template_surface,
                exclude_subjects=subject_subset_exclude,
            ),
        )

        # =====================================================================
        # MODEL
        # =====================================================================

        UNET_DECODER_CHANNELS_POST = {
            ("t1w", "1mm"):         None,
            ("synth", "1mm"):       UNET_DECODER_CHANNELS["t1w","1mm"],
            ("synth", "random"):    UNET_DECODER_CHANNELS["t1w","1mm"],
        }

        unet = body.UNet(
            spatial_dims=3,
            in_channels=1,
            encoder_channels=UNET_ENCODER_CHANNELS[contrast, resolution],
            decoder_channels=UNET_DECODER_CHANNELS[contrast, resolution],
            return_encoder_features=UNET_RETURN_ENCODER_FEATURES,
            return_decoder_features=UNET_RETURN_DECODER_FEATURES,
            # match the synth features to the T1w features
            encoder_post=None,
            decoder_post=UNET_DECODER_CHANNELS_POST[contrast, resolution],
        )

        topofit = head.TopoFit(
            in_channels=unet.num_features,
            in_order=TOPOFIT_ORDER_IN,
            out_order=TOPOFIT_ORDER_OUT,
            max_order=TOPOFIT_ORDER_MAX,
            white_feature_maps=[unet.decoder_features] * (TOPOFIT_ORDER_MAX - TOPOFIT_ORDER_IN + 1),
            white_channels=TOPOFIT_WHITE_MATTER_CHANNELS,
            pial_feature_maps=unet.decoder_features,
            pial_channels=TOPOFIT_GRAY_MATTER_CHANNELS,
            pial_deform_module=TOPOFIT_GRAY_MATTER_MODULE,
            device=device
        )

        self.model = config.BrainNetParameters(
            device=device,
            body=unet,
            heads=dict(surface=topofit),
        )

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
            load_body_from_checkpoint=load_body_from_checkpoint,
            load_head_from_checkpoint=load_head_from_checkpoint,
        )

        # =====================================================================
        # SYNTHESIZER
        # =====================================================================

        self.synthesizer = config.SynthesizerParameters(
            train=brainsynth.config.SynthesizerConfig(
                builder=builder_train,
                out_size=fov_out_size,
                out_center_str=fov_out_center_str,
                # segmentation_labels = "brainseg"
                # photo_mode = False
                # photo_spacing_range = [2.0, 7.0]
                # photo_thickness = 0.001
                selectable_images=images_train_sel,
                device=device,
            ),
            validation=brainsynth.config.SynthesizerConfig(
                builder=builder_validation,
                out_size=fov_out_size,
                out_center_str=fov_out_center_str,
                # segmentation_labels = "brainseg"
                # photo_mode = False
                # photo_spacing_range = [2.0, 7.0]
                # photo_thickness = 0.001
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
