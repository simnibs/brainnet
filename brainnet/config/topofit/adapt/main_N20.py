from pathlib import Path

import torch

import brainsynth.config

from ignite.engine import Events

from brainnet import config
from brainnet.modules import body, head
# Parameters defined in other files
from . import events_evaluator, events_trainer, losses

# =============================================================================
# GENERAL VARIABLES
# =============================================================================

mode_contrast = "t1w"     # synth, t1w, t2w, flair
mode_resolution = "1mm"     # 1mm, random

project: str = "TopoFit"
run: str = f"01_lh_{mode_contrast}-adapt-sourceonly_{mode_resolution}"

run_id: None | str = None
# NOTE Resume from SYNTH run!
resume_from_run: None | str = f"01_lh_synth_{mode_resolution}"
resume_from_run = None
tags = [mode_contrast, mode_resolution, "adapt:t1w"]
device: str | torch.device  = torch.device("cuda:0")

gradient_accumulation_steps = 1

target_surface_resolution: int = 5
target_surface_hemisphere: str = "lh"
initial_surface_resolution: int = 0

out_size = [128, 224, 160]
out_center_str = "lh"

out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")

scratch_root_dir: Path = Path("/mnt/scratch/personal/jesperdn/training_data")
train_data_dir = scratch_root_dir / "pseudo20"

projects_root_dir: Path = Path("/mnt/projects/CORTECH/nobackup/training_data")
validation_data_dir = projects_root_dir / "full"

datasets = None # ["HCP"]

# =============================================================================
# TRAINING MODE
# =============================================================================

match mode_contrast:
    case "synth":
        images_train = ["generation_labels_dist"]
        images_val = ["t1w"]
        subject_subset_train = "train"
        subject_subset_val = "validation"
        datasets = datasets
    case "t1w":
        images_train = ["t1w"]
        images_val = ["t1w"]
        subject_subset_train = "source20" # "train"
        subject_subset_val = "validation"
        datasets = datasets
    case "t2w":
        images_train = ["t2w"]
        images_val = ["t2w"]
        # HCP sub-059 excluded: T2w is just zeros!
        subject_subset_train = "train.t2"
        subject_subset_val = "validation.t2"
        datasets = ["HCP", "OASIS3"] if datasets is None else datasets
    case "flair":
        images_train = ["flair"]
        images_val = ["flair"]
        subject_subset_train = "train.flair"
        subject_subset_val = "validation.flair"
        datasets = ["ADNI3", "AIBL"] if datasets is None else datasets
    case _:
        raise ValueError

match mode_resolution:
    case "1mm":
        builder_res = "Iso"
    case "random":
        builder_res = ""
    case _:
        raise ValueError

builder_contrast = "Synth" if mode_contrast == "synth" else "Select"
builder_train = f"Only{builder_contrast}{builder_res}"
builder_validation = f"OnlySelect{builder_res}"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

cfg_train = config.TrainParameters(
    max_epochs = 5000,
    epoch_length_train = 100,
    epoch_length_val = 50,
    gradient_accumulation_steps = gradient_accumulation_steps,
    evaluate_on=Events.EPOCH_COMPLETED(every=10),
    events_trainer = events_trainer.events,
    events_evaluators = events_evaluator.events,
    enable_amp = True,
)

# =============================================================================
# DATALOADER
# =============================================================================

cfg_dataloader = config.DataloaderParameters()

# =============================================================================
# DATASETS
# =============================================================================

cfg_dataset = config.DatasetParameters(
    train = brainsynth.config.DatasetConfig(
        root_dir = train_data_dir,
        subject_dir = scratch_root_dir, #projects_root_dir / "subject_splits",
        subject_subset = subject_subset_train,
        datasets = datasets,
        images = images_train,
        target_surface_resolution = target_surface_resolution,
        target_surface_hemispheres = target_surface_hemisphere,
        initial_surface_resolution = initial_surface_resolution,
    ),
    validation = brainsynth.config.DatasetConfig(
        root_dir = validation_data_dir,
        subject_dir = projects_root_dir / "subject_splits",
        subject_subset = subject_subset_val,
        datasets = datasets,
        images = images_val,
        target_surface_resolution = target_surface_resolution,
        target_surface_hemispheres = target_surface_hemisphere,
        initial_surface_resolution = initial_surface_resolution,
    ),
)


# =============================================================================
# CRITERION
# =============================================================================

cfg_criterion = config.CriterionParameters(
    train=losses.cfg_loss,
    validation=losses.cfg_loss, # could/should be different...
)

# =============================================================================
# MODEL
# =============================================================================

spatial_dims = 3
in_channels = 1

unet_enc_ch = [[16], [32], [64], [96], [128]]
unet_dec_ch = [[96], [64], [64], [32]]
unet_encoder_features = [True, True, True, True, False]
unet_decoder_features = [True, True, True, True]

unet_out_ch = [i[0] for i,j in zip(
    unet_enc_ch + unet_dec_ch, unet_encoder_features + unet_decoder_features
) if j]

unet = body.UNet(
    spatial_dims,
    in_channels,
    unet_enc_ch,
    unet_dec_ch,
    return_encoder_features=unet_encoder_features,
    return_decoder_features=unet_decoder_features,
)

cfg_model = config.BrainNetParameters(
    device=device,
    body = unet,
    heads = dict(
        surface = head.TopoFit(
            in_channels=unet.num_features,
            out_res=target_surface_resolution,
            device=device,
        ),
    ),
)

# =============================================================================
# OPTIMIZER
# =============================================================================

cfg_optimizer = config.OptimizerParameters("AdamW", dict(lr=1.0e-4))

# =============================================================================
# RESULTS
# =============================================================================

cfg_results = config.ResultsParameters(
    out_dir=out_dir / project / run,
    load_from_dir = out_dir / project / resume_from_run if resume_from_run is not None else None,
)

# =============================================================================
# SYNTHESIZER
# =============================================================================

cfg_synth = config.SynthesizerParameters(
    train=brainsynth.config.SynthesizerConfig(
        builder = builder_train,
        out_size = out_size,
        out_center_str = out_center_str,
        # segmentation_labels = "brainseg"
        # photo_mode = False
        # photo_spacing_range = [2.0, 7.0]
        # photo_thickness = 0.001
        alternative_images = images_train, # ["t1w", "t2w", "flair"],
        device = device,
    ),
    validation=brainsynth.config.SynthesizerConfig(
        builder = builder_validation,
        out_size = out_size,
        out_center_str = out_center_str,
        # segmentation_labels = "brainseg"
        # photo_mode = False
        # photo_spacing_range = [2.0, 7.0]
        # photo_thickness = 0.001
        alternative_images = images_val, # ["t1w", "t2w", "flair"],
        device = device,
    ),
)

# =============================================================================
# WANDB
# =============================================================================

cfg_wandb = config.WandbParameters(
    enable=True,
    project=project,
    name=run,
    wandb_dir=out_dir / "wandb",
    log_on = cfg_train.evaluate_on,
    run_id = run_id,
    tags=tags,
)


train_setup = config.TrainSetup(
    project,
    run,
    device,
    cfg_criterion,
    cfg_dataloader,
    cfg_dataset,
    cfg_model,
    cfg_optimizer,
    cfg_results,
    cfg_synth,
    cfg_train,
    cfg_wandb,
)
