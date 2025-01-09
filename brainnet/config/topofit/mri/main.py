from pathlib import Path

import torch

import brainsynth.config

from brainnet import config
from brainnet.modules import body, head
# Parameters defined in other files
from . import events_evaluator, events_trainer, losses

"""

# Stage 1:
# - surface resolution 4
# - add chamfer at 500
python brainnet/train/brainnet_train.py brainnet.config.topofit.mri.main --max-epochs 800

# Stage 2:
# - increase target surface resolution to 5
# - decrease LR by factor 0.5
python brainnet/train/brainnet_train.py brainnet.config.topofit.mri.main --load-checkpoint 800 --max-epochs 1400

# Stage 3:
# - increase target surface resolution to 6
python brainnet/train/brainnet_train.py brainnet.config.topofit.mri.main --load-checkpoint 1400 --max-epochs 1600

"""

# =============================================================================
# GENERAL VARIABLES
# =============================================================================

mode_contrast = "t2w"     # synth, t1w, t2w, flair
mode_resolution = "1mm"     # 1mm, random

project: str = "TopoFit"
run: str = f"{mode_contrast}_{mode_resolution}"

run_id: None | str = None # f"{run}-00"
resume_from_run: None | str = None # run
tags = [mode_contrast, mode_resolution]
device: str | torch.device  = torch.device("cuda:0")

resolution = 6
target_surface = dict(resolution=resolution, name="target-decoupled")

# Single hemisphere
# target_surface_hemisphere: str = "lh"
# out_size = [128, 224, 160]
# out_center_str = "lh"

# Full brain
out_size = [176, 208, 176]
out_center_str = "brain"

root_dir: Path = Path("/mnt/projects/CORTECH/nobackup/training_data")
out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")

# =============================================================================
# TRAINING MODE
# =============================================================================

# Use COBRE and MCIC as test sets
# ISBI2015 are not great from FS
datasets = [
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

match mode_contrast:
    case "synth":
        images_train = ["generation_labels_dist", "t1w"]
        images_val = ["t1w"]
        subject_subset_train = "train"
        subject_subset_val = "validation"
    case "t1w":
        images_train = ["t1w"]
        images_val = ["t1w"]
        subject_subset_train = "train"
        subject_subset_val = "validation"
    case "t2w":
        images_train = ["t2w"]
        images_val = ["t2w"]
        # HCP sub-059 excluded: T2w is just zeros!
        subject_subset_train = "train.t2"
        subject_subset_val = "validation.t2"
        datasets = ["HCP", "OASIS3"]
    case "flair":
        images_train = ["flair"]
        images_val = ["flair"]
        subject_subset_train = "train.flair"
        subject_subset_val = "validation.flair"
        datasets = ["ADNI3", "AIBL"]
    case _:
        raise ValueError

match mode_resolution:
    case "1mm":
        builder_res = "Iso"
    case "random":
        builder_res = ""
    case _:
        raise ValueError

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

cfg_train = config.TrainParameters(
    max_epochs = 5000,
    epoch_length_train = 100,
    epoch_length_val = 50,
    gradient_accumulation_steps = 1,
    events_trainer = events_trainer.events,
    events_evaluators = events_evaluator.events,
    enable_amp = True,
)

# =============================================================================
# DATALOADER
# =============================================================================

builder_contrast = "Synth" if mode_contrast == "synth" else "Select"
builder_train = f"Only{builder_contrast}{builder_res}"
builder_validation = f"OnlySelect{builder_res}"

cfg_dataloader = config.DataloaderParameters()

# =============================================================================
# DATASETS
# =============================================================================

cfg_dataset = config.DatasetParameters(
    train = brainsynth.config.DatasetConfig(
        root_dir = root_dir / "full",
        subject_dir = root_dir / "subject_splits",
        subject_subset = subject_subset_train,
        datasets = datasets,
        images = images_train,
        target_surface = target_surface,
    ),
    validation = brainsynth.config.DatasetConfig(
        root_dir = root_dir / "full",
        subject_dir = root_dir / "subject_splits",
        subject_subset = subject_subset_val,
        datasets = datasets,
        images = images_val,
        target_surface = target_surface,
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
            out_res=resolution,
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
