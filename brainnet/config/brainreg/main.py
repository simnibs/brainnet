from pathlib import Path

import torch

import brainsynth.config

from brainnet import config
from brainnet.modules import body, head
from brainsynth.constants import SURFACE

# Parameters defined in other files
from brainnet.config.brainreg import events_trainer, events_evaluator
from brainnet.config.brainreg.losses import cfg_loss


# =============================================================================
# GENERAL VARIABLES
# =============================================================================

project: str = "BrainReg"
run: str = "18-run-image"
# run: str = "run-image-15"

run_id: None | str = None  # f"{run}-00"
resume_from_run: None | str = None  # "run-image-11"  # run
tags = ["t1w", "deep features"]
device: str | torch.device = torch.device("cuda")

target_surface_resolution: int | None = 5
target_surface_hemisphere: str = "both"
initial_surface_resolution = None

root_dir: Path = Path("/mnt/projects/CORTECH/nobackup/training_data")
out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

cfg_train = config.TrainParameters(
    max_epochs=3000,
    epoch_length_train=100,
    gradient_accumulation_steps=1,
    epoch_length_val=25,
    events_trainer=events_trainer.events,
    events_evaluators=events_evaluator.events,
    enable_amp=True,
)

# =============================================================================
# DATALOADER
# =============================================================================

# reshaped to channels! so we need to drop last batch if incomplete!
cfg_dataloader = config.DataloaderParameters(batch_size=2, drop_last=True)

# =============================================================================
# DATASETS
# =============================================================================

cfg_dataset = config.DatasetParameters(
    train=brainsynth.config.DatasetConfig(
        root_dir=root_dir / "brainreg",
        subject_dir=root_dir / "subject_splits",
        subject_subset="train.registration",
        images=["t1w_areg_mni", "brainseg_with_extracerebral"],
        load_mask="force",
        target_surface_resolution=target_surface_resolution,
        target_surface_hemispheres=target_surface_hemisphere,
        target_surface_files=SURFACE.files.target,
        initial_surface_resolution=initial_surface_resolution,
    ),
    validation=brainsynth.config.DatasetConfig(
        root_dir=root_dir / "brainreg",
        subject_dir=root_dir / "subject_splits",
        subject_subset="validation.registration",
        images=["t1w_areg_mni", "brainseg_with_extracerebral"],
        load_mask="force",
        target_surface_resolution=target_surface_resolution,
        target_surface_hemispheres=target_surface_hemisphere,
        target_surface_files=SURFACE.files.target,
        initial_surface_resolution=initial_surface_resolution,
    ),
)

# =============================================================================
# CRITERION
# =============================================================================

cfg_criterion = config.CriterionParameters(
    train=cfg_loss,
    validation=cfg_loss,  # could/should be different...
)

# =============================================================================
# MODEL
# =============================================================================

spatial_dims = 3
in_channels = 2

# unet_enc_ch = [[8], [16], [32], [64], [128]]
# unet_dec_ch = [[64], [32], [16], [16]]
# unet_decoder_features = [True, True, True, True]
# use_feature_maps = ["decoder:0", "decoder:1", "decoder:2", "decoder:3"]

unet_enc_ch = [[32], [64], [96], [128], [160]]
unet_dec_ch = [[128], [96], [64], [32]]
unet_encoder_features = [True, True, True, True, False]
unet_decoder_features = [True, True, True, True]
unet = body.UNet(
    spatial_dims,
    in_channels,
    unet_enc_ch,
    unet_dec_ch,
    return_encoder_features=unet_encoder_features,
    return_decoder_features=unet_decoder_features,
)
use_feature_maps = [
    ["encoder:3", "decoder:0"],
    ["encoder:2", "decoder:1"],
    ["encoder:1", "decoder:2"],
    ["encoder:0", "decoder:3"],
]

intermediate = 64

svf_modules = torch.nn.ModuleList(
    [
        head.SVFModule(
            [
                sum([unet.num_features[fmap] for fmap in fmaps]),
                intermediate,
                intermediate,
                3,
            ],
            fmaps,
        ) for fmaps in use_feature_maps
    ]
)

cfg_model = config.BrainRegParameters(
    device=device,
    body=unet,
    svf=svf_modules,
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
    load_from_dir=(
        out_dir / project / resume_from_run if resume_from_run is not None else None
    ),
    examples_keys=["t1w_areg_mni", "surface"],
)

# =============================================================================
# SYNTHESIZER
# =============================================================================

cfg_synth = config.SynthesizerParameters(train=None, validation=None)

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
