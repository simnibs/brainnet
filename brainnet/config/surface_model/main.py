from pathlib import Path

from ignite.engine import Events
import torch

import brainsynth.config

from brainnet import config
from brainnet.modules import body, head
# Parameters defined in other files
from brainnet.config.surface_model import events_trainer, events_evaluators
from .losses import cfg_loss

# =============================================================================
# GENERAL VARIABLES
# =============================================================================

project: str = "BrainNet"
run: str = "lh_1mm-iso_PV_really-1mm"
run_id: None | str = None # f"{run}-00"
resume_from_run: None | str = None # run
tags = ["synth", "1mm res", "rand PV"]
device: str | torch.device  = torch.device("cuda:0")

target_surface_resolution: int = 5
target_surface_hemisphere: str = "lh"
initial_surface_resolution: int = 0

root_dir: Path = Path("/mnt/projects/CORTECH/nobackup/")
out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

cfg_train = config.TrainParameters(
    max_epochs = 5000,
    epoch_length_train=100,
    epoch_length_val=25,
    # evaluate_on=Events.EPOCH_COMPLETED,
    events_trainer=events_trainer.events,
    events_evaluators=events_evaluators.events,
    enable_amp=True,
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
        root_dir = root_dir / "training_data",
        subject_dir = root_dir / "training_data_subjects",
        subject_subset = "train",
        images = ["generation_labels_dist"],
        # images = ["t1w"],
        target_surface_resolution = target_surface_resolution,
        target_surface_hemispheres = target_surface_hemisphere,
        initial_surface_resolution = initial_surface_resolution,
    ),
    validation = brainsynth.config.DatasetConfig(
        root_dir = root_dir / "training_data",
        subject_dir = root_dir / "training_data_subjects",
        subject_subset = "validation",
        images = ["t1w"],
        target_surface_resolution = target_surface_resolution,
        target_surface_hemispheres = target_surface_hemisphere,
        initial_surface_resolution = initial_surface_resolution,
    ),
)

# # T2w
# # HCP sub-059 excluded: T2w is just zeros!
# cfg_dataset = config.DatasetParameters(
#     train = brainsynth.config.DatasetConfig(
#         root_dir = root_dir / "training_data",
#         subject_dir = root_dir / "training_data_subjects",
#         subject_subset = "train.t2",
#         datasets = ["HCP", "OASIS3"],
#         images = ["t2w"],
#         target_surface_resolution = target_surface_resolution,
#         target_surface_hemispheres = target_surface_hemisphere,
#         initial_surface_resolution = initial_surface_resolution,
#     ),
#     validation = brainsynth.config.DatasetConfig(
#         root_dir = root_dir / "training_data",
#         subject_dir = root_dir / "training_data_subjects",
#         subject_subset = "validation.t2",
#         datasets = ["HCP", "OASIS3"],
#         images = ["t2w"],
#         target_surface_resolution = target_surface_resolution,
#         target_surface_hemispheres = target_surface_hemisphere,
#         initial_surface_resolution = initial_surface_resolution,
#     ),
# )

# # FLAIR
# cfg_dataset = config.DatasetParameters(
#     train = brainsynth.config.DatasetConfig(
#         root_dir = root_dir / "training_data",
#         subject_dir = root_dir / "training_data_subjects",
#         subject_subset = "train.flair",
#         datasets = ["ADNI3", "AIBL"],
#         images = ["flair"],
#         target_surface_resolution = target_surface_resolution,
#         target_surface_hemispheres = target_surface_hemisphere,
#         initial_surface_resolution = initial_surface_resolution,
#     ),
#     validation = brainsynth.config.DatasetConfig(
#         root_dir = root_dir / "training_data",
#         subject_dir = root_dir / "training_data_subjects",
#         subject_subset = "validation.flair",
#         datasets = ["ADNI3", "AIBL"],
#         images = ["flair"],
#         target_surface_resolution = target_surface_resolution,
#         target_surface_hemispheres = target_surface_hemisphere,
#         initial_surface_resolution = initial_surface_resolution,
#     ),
# )


# =============================================================================
# CRITERION
# =============================================================================

cfg_criterion = config.CriterionParameters(
    train=cfg_loss,
    validation=cfg_loss, # could/should be different...
)

# =============================================================================
# MODEL
# =============================================================================

spatial_dims = 3
in_channels = 1
unet_enc_ch = [[32], [64], [96], [128], [160]]
unet_dec_ch = [[128], [96], [64], [64]]
unet_out_ch = unet_dec_ch[-1][-1]

cfg_model = config.BrainNetParameters(
    device=device,
    body = body.UNet(spatial_dims, in_channels, unet_enc_ch, unet_dec_ch),
    heads = dict(
        surface = head.SurfaceModule(
            in_channels=unet_out_ch,
            prediction_res=target_surface_resolution,
            device=device,
        ),
#                 kwargs = dict(prediction_res = target_surface_resolution),

        # segmentation = SegmentationModule(...)
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

out_size = [128, 224, 160]
out_center_str = "lh"

cfg_synth = config.SynthesizerParameters(
    train=brainsynth.config.SynthesizerConfig(
        builder = "OnlySynthIso",
        # builder = "OnlySelect",
        # in_res = [1.0, 1.0, 1.0]
        out_size = out_size,
        out_center_str = out_center_str,
        # segmentation_labels = "brainseg"
        # photo_mode = False
        # photo_spacing_range = [2.0, 7.0]
        # photo_thickness = 0.001
        alternative_images = ["t1w"], # ["t1w", "t2w", "flair"],
        device = device,
    ),
    validation=brainsynth.config.SynthesizerConfig(
        builder = "OnlySelectIso",
        out_size = out_size,
        out_center_str = out_center_str,
        # segmentation_labels = "brainseg"
        # photo_mode = False
        # photo_spacing_range = [2.0, 7.0]
        # photo_thickness = 0.001
        alternative_images = ["t1w"], # ["t1w", "t2w", "flair"],
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
