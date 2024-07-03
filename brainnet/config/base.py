from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from ignite.engine.events import CallableEventWithFilter
from ignite.engine import Events

from brainsynth.config import DatasetConfig, SynthesizerConfig
from brainnet.modules.loss_wrappers import RegularizationLoss, SupervisedLoss


@dataclass
class LossParameters:
    functions: dict[str, dict[str, RegularizationLoss | SupervisedLoss]]
    head_weights: dict[str, float]
    loss_weights: dict[str, dict[str, float]]


@dataclass
class CriterionParameters:
    train: LossParameters
    validation: LossParameters


@dataclass
class DataloaderParameters:
    batch_size: int = 1
    num_workers: int = 4
    prefetch_factor: int = 2
    # kwargs: dict | None = None

    # def __post_init__(self):
    #     if self.kwargs is None:
    #         self.kwargs = {}

@dataclass
class DatasetParameters:
    train: DatasetConfig
    validation: DatasetConfig


@dataclass
class EventAction:
    event: CallableEventWithFilter
    handler: Callable
    kwargs: dict | None # kwargs passed to handler (besides engine)


@dataclass
class ModelParameters:
    """Filename is only used if config is not specified."""
    device: torch.device
    body: torch.nn.Module
    heads: dict[str, torch.nn.Module]


@dataclass
class OptimizerParameters:
    name: str
    kwargs: dict
    lr_parameter_groups: dict | None = None

    # lr_factor: 0.5 # applied after optimizer state is loaded

    # lr_parameter_groups:
        # body: 1.0e-4
        # heads:
        # surface: 1.0e-4
        # segmentation: 1.0e-3


@dataclass
class ResultsParameters:
    out_dir: Path | str
    checkpoint_prefix: str = "ckpt"
    checkpoint_subdir: str = "checkpoint"
    examples_subdir: str = "examples"
    save_checkpoint_on: CallableEventWithFilter = Events.EPOCH_COMPLETED(every=20)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.checkpoint_dir = self.out_dir / self.checkpoint_subdir
        self.examples_dir = self.out_dir / self.examples_subdir

        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)
        if not self.examples_dir.exists():
            self.examples_dir.mkdir(parents=True)

@dataclass
class SynthesizerParameters:
    train: None | SynthesizerConfig = None
    validation: None | SynthesizerConfig = None


@dataclass
class TrainParameters:
    max_epochs: int
    resume_from_checkpoint: int
    epoch_length_train: int = 100
    epoch_length_val: int = 50
    evaluate_on: CallableEventWithFilter = Events.EPOCH_COMPLETED(every=10)
    events: list[EventAction] | None = None

    enable_amp: bool = True

    # Estimate inter-task affinity (ITA)
    enable_ITA_estimation: bool = False

    # Consider the model "converged" when the LR is reduced to this value
    minimum_lr: float = 1e-10

    surface_decoupling_amount: float = 0.2 # 0.0 to disable


@dataclass
class WandbParameters:
    # Logging using Weights & Biases (https://wandb.ai)
    enable: bool
    project: str
    name: str
    wandb_dir: Path | str
    log_on: CallableEventWithFilter
    #resume: str = "auto"

    # kwargs: dict | None = None

    # def __post_init__(self):
    #     if self.kwargs is None:
    #         self.kwargs = {}


@dataclass
class TrainSetup:
    """Class to collect all training related configurations."""
    project: str
    run: str
    device: str | torch.device

    criterion: CriterionParameters
    dataloader: DataloaderParameters
    dataset: DatasetParameters
    model: ModelParameters
    optimizer: OptimizerParameters
    results: ResultsParameters
    synthesizer: SynthesizerParameters
    train_params: TrainParameters
    wandb: WandbParameters
