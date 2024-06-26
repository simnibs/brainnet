from pathlib import Path

import torch

from brainnet import config
from brainnet.modules import body, head

# Parameters defined in other files
from .events import events
from .losses import cfg_loss

# =============================================================================
# GENERAL VARIABLES
# =============================================================================

project: str = "BrainNet"
run: str = "lh-01"
device: str | torch.device  = torch.device("cuda:0")

target_surface_resolution: int = 5
target_surface_hemisphere: str = "lh"
initial_surface_resolution: int = 0

root_dir: Path = Path("/mnt/projects/CORTECH/nobackup/")
out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")

# =============================================================================
# EVENTS
# =============================================================================
#
# Actions to take at certain points duing the training, e.g., modify loss
# weights

# events = ...

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

cfg_train = config.TrainParameters(
    max_epochs = 2000,
    resume_from_checkpoint = 0,
    events=events,
    surface_decoupling_amount = 0.2, # 0.0 to disable
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
        # synthesizer = SynthesizerConfig( ... ),
        # datasets = ["ABIDE", "HCP"], # default: all
        # images = ["generation_labels", "t1w"],
        # ds_structure = "flat",
        target_surface_resolution = target_surface_resolution,
        target_surface_hemispheres = target_surface_hemisphere,
        initial_surface_resolution = initial_surface_resolution,
    ),
    validation = brainsynth.config.DatasetConfig(
        root_dir = root_dir / "training_data",
        subject_dir = root_dir / "training_data_subjects",
        subject_subset = "validation",
        target_surface_resolution = target_surface_resolution,
        target_surface_hemispheres = target_surface_hemisphere,
        initial_surface_resolution = initial_surface_resolution,
    ),
)


# =============================================================================
# MODEL
# =============================================================================

# cfg_loss = ...

# =============================================================================
# MODEL
# =============================================================================

spatial_dims = 3
in_channels = 1
unet_enc_ch = [[32], [64], [96], [128], [160]]
unet_dec_ch = [[128], [96], [64], [64]]
unet_out_ch = unet_dec_ch[-1][-1]

cfg_model = config.ModelParameters(
    device=device,
    body = body.UNet(spatial_dims, in_channels, unet_enc_ch, unet_dec_ch),
    heads = dict(
        surface = head.SurfaceModule(
            in_channels=unet_out_ch,
            prediction_res=target_surface_resolution,
            device=device,
        ),
        # segmentation = SegmentationModule(...)
    ),
)

# model_conf = config.ModelParameters(
#     device = device,
#     body = dict(
#         model = "UNet",
#         kwargs = dict(
#             spatial_dims = 3,
#             in_channels = 1,
#             encoder_channels = [[32], [64], [96], [128], [160]],
#             decoder_channels = [[128], [96], [64], [64]],
#         )
#     ),
#     heads = dict(
#         surface = dict(
#             module = dict(
#                 name = "SurfaceModule",
#                 kwargs = dict(prediction_res = target_surface_resolution),
#             ),
#             runtime_kwargs = dict(return_pial = True),
#         ),
#     ),
# )

# =============================================================================
# OPTIMIZER
# =============================================================================

cfg_optimizer = config.OptimizerParameters("AdamW", dict(lr=1.0e-4))


# =============================================================================
# RESULTS
# =============================================================================

cfg_results = config.ResultsParameters(out_dir=out_dir / project / run)

# =============================================================================
# SYNTHESIZER
# =============================================================================

cfg_synth = SynthesizerParameters(
    train=SynthesizerConfig(
        builder = "DefaultSynthBuilder",
        # in_res = [1.0, 1.0, 1.0]
        out_size = [192, 192, 192],
        # align_corners = True
        out_center_str = "lh",
        # segmentation_labels = "brainseg"
        # photo_mode = False
        # photo_spacing_range = [2.0, 7.0]
        # photo_thickness = 0.001
        # alternative_images = [t1w, t2w]
        device = device,
    ),
    validation=SynthesizerConfig(
        builder = "ValidationSynthBuilder",
        # in_res = [1.0, 1.0, 1.0]
        out_size = [192, 192, 192],
        # align_corners = True
        out_center_str = "lh",
        # segmentation_labels = "brainseg"
        # photo_mode = False
        # photo_spacing_range = [2.0, 7.0]
        # photo_thickness = 0.001
        # alternative_images = [t1w, t2w]
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
    # kwargs = dict(
    # tags =,
    # entity = None,  # username/team name to send data to
    # )
)


train_setup = config.TrainSetup(
    project,
    run,
    device,
    cfg_dataloader,
    cfg_dataset,
    cfg_loss,
    cfg_model,
    cfg_optimizer,
    cfg_results,
    cfg_synth,
    cfg_train,
    cfg_wandb,
)
