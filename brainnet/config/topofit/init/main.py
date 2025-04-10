from pathlib import Path

import torch
from ignite.engine import Events

import brainsynth.config
from brainnet import config
from brainnet.modules import body, head

# Parameters defined in other files
from . import events_evaluator, events_trainer, losses

"""

# Stage 1:
# - surface resolution 6
# - add chamfer at 500
python brainnet/train/brainnet_train.py brainnet.config.topofit.mri.main --max-epochs 800

python brainnet/train/brainnet_train.py brainnet.config.topofit.mri.main --load-checkpoint 500 --max-epochs 1000

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

mode_contrast = "synth"  # synth, t1w, t2w, flair
mode_resolution = "random"  # 1mm, random
tags = []

project: str = "TopoInit"
run: str = f"{mode_contrast}_{mode_resolution}-01"

run_id: None | str = None  # f"{run}-00"
resume_from_run: None | str = run  # None # run
tags += [mode_contrast, mode_resolution]
device: str | torch.device = torch.device("cuda:0")

in_order = 3
out_order = 5
max_order = 5
template_surface = dict(resolution=in_order, name="template")
target_vertices = dict(resolution=out_order, name="target", types=["white"])

# Single hemisphere
# target_surface_hemisphere: str = "lh"
# out_size = [128, 224, 160]
# out_center_str = "lh"

# Full brain
out_size = [176, 208, 176]
out_center_str = "brain"

random_skullstrip = True

data_dir: Path = Path("/mnt/projects/CORTECH/nobackup/training_data")
out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")
model_dir = out_dir
# model_dir: Path = Path("/mnt/projects/CORTECH/nobackup/jesper/models")

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

# {ds}.exclude.txt
subject_subset_exclude = "exclude"

match mode_contrast:
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
    max_epochs=5000,
    epoch_length_train=100,
    epoch_length_val=50,
    gradient_accumulation_steps=1,
    # evaluate_on=Events.EPOCH_COMPLETED(every=1),
    events_trainer=events_trainer.events,
    events_evaluators=events_evaluator.events,
    enable_amp=True,
)

# =============================================================================
# DATALOADER
# =============================================================================

builder_contrast = "Synth" if mode_contrast == "synth" else "Select"
if builder_contrast == "Synth" or random_skullstrip:
    # synth has skullstrip anyway
    builder_train = f"Only{builder_contrast}{builder_res}"
else:
    builder_train = f"Only{builder_contrast}NoSkullStrip{builder_res}"
builder_validation = f"OnlySelectNoSkullStrip{builder_res}"

cfg_dataloader = config.DataloaderParameters()

# =============================================================================
# DATASETS
# =============================================================================

cfg_dataset = config.DatasetParameters(
    train=brainsynth.config.DatasetConfig(
        root_dir=data_dir / "full",
        subject_dir=data_dir / "subject_splits",
        subject_subset=subject_subset_train,
        datasets=datasets,
        images=images_train,
        target_vertices=target_vertices,
        template_surface=template_surface,
        exclude_subjects=subject_subset_exclude,
    ),
    validation=brainsynth.config.DatasetConfig(
        root_dir=data_dir / "full",
        subject_dir=data_dir / "subject_splits",
        subject_subset=subject_subset_val,
        datasets=datasets,
        images=images_val,
        target_vertices=target_vertices,
        template_surface=template_surface,
        exclude_subjects=subject_subset_exclude,
    ),
)


# =============================================================================
# CRITERION
# =============================================================================

cfg_criterion = config.CriterionParameters(
    train=losses.cfg_loss,
    validation=losses.cfg_loss,  # could/should be different...
)

medial_wall_weights = None
# medial_wall_weights = (1.0, 0.1) # non-MD/MD

# =============================================================================
# MODEL
# =============================================================================

unet = body.UNet(
    spatial_dims=3,
    in_channels=1,
    encoder_channels=[[8], [16], [32], [64], [128]],
    decoder_channels=[[64], [32], [16], [8]],
    return_encoder_features=None,
    return_decoder_features=[True, True, True, True],
    encoder_post=None,
    decoder_post=None,
)

topoinit = head.TopoInit(
    in_channels=unet.num_features,
    in_order=in_order,
    out_order=out_order,
    max_order=max_order,
    feature_maps=[
        unet.decoder_features,
        unet.decoder_features,
        unet.decoder_features,
    ],
    channels=dict(
        encoder=[64,64,64,64],
        decoder=[64,64,64],
    ),
    device=device,
)

cfg_model = config.BrainNetParameters(
    device=device,
    body=unet,
    heads=dict(surface=topoinit),
)

# =============================================================================
# OPTIMIZER
# =============================================================================

cfg_optimizer = config.OptimizerParameters("AdamW", dict(lr=1.0e-3))

# =============================================================================
# RESULTS
# =============================================================================

cfg_results = config.ResultsParameters(
    out_dir=out_dir / project / run,
    load_from_dir=model_dir / project / resume_from_run
    if resume_from_run is not None
    else None,
    # save_example_on=Events.EPOCH_COMPLETED(every=1),
)

# =============================================================================
# SYNTHESIZER
# =============================================================================

cfg_synth = config.SynthesizerParameters(
    train=brainsynth.config.SynthesizerConfig(
        builder=builder_train,
        out_size=out_size,
        out_center_str=out_center_str,
        # segmentation_labels = "brainseg"
        # photo_mode = False
        # photo_spacing_range = [2.0, 7.0]
        # photo_thickness = 0.001
        selectable_images=images_train_sel,
        device=device,
    ),
    validation=brainsynth.config.SynthesizerConfig(
        builder=builder_validation,
        out_size=out_size,
        out_center_str=out_center_str,
        # segmentation_labels = "brainseg"
        # photo_mode = False
        # photo_spacing_range = [2.0, 7.0]
        # photo_thickness = 0.001
        selectable_images=images_val_sel,  # ["t1w", "t2w", "flair"],
        device=device,
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
    log_on=cfg_train.evaluate_on,
    run_id=run_id,
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
