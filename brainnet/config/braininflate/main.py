from pathlib import Path

from ignite.engine import Events
import torch

import brainsynth.config

from brainnet import config
from brainnet.modules import BrainInflate

# Parameters defined in other files
from . import events_trainer, events_evaluator
from .losses import cfg_loss

# =============================================================================
# GENERAL VARIABLES
# =============================================================================

project: str = "BrainInflate"
run: str = "01"

tags = []

run_id: None | str = None
resume_from_run: None | str = None
device: str | torch.device = torch.device("cuda:0")

initial_surface = dict(types="white", resolution=6, name="prediction")
target_surface = dict(types="inflated", resolution=None, name=None)

root_dir: Path = Path("/mnt/projects/CORTECH/nobackup/training_data")
out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")

# =============================================================================
# TRAINING MODE
# =============================================================================

subject_subset_train = "train"
subject_subset_val = "validation"
datasets = None

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

cfg_train = config.TrainParameters(
    max_epochs = 5000,
    epoch_length_train = 100,
    epoch_length_val = 25,
    gradient_accumulation_steps = 1,
    # evaluate_on=Events.EPOCH_COMPLETED,
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
        root_dir = root_dir / "spherereg",
        subject_dir = root_dir / "subject_splits",
        subject_subset = subject_subset_train,
        datasets = datasets,
        images = [],
        target_surface = target_surface,
        initial_surface = initial_surface,
    ),
    validation = brainsynth.config.DatasetConfig(
        root_dir = root_dir / "spherereg",
        subject_dir = root_dir / "subject_splits",
        subject_subset = subject_subset_val,
        datasets = datasets,
        images = [],
        target_surface = target_surface,
        initial_surface = initial_surface,
    ),
)


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

model = BrainInflate(out_channels=3, n_steps=10, device=device)
model.to(device)

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
    model,
    cfg_optimizer,
    cfg_results,
    None,
    cfg_train,
    cfg_wandb,
)
