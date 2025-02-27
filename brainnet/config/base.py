from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
    drop_last: bool = False
    # kwargs: dict | None = None

    # def __post_init__(self):
    #     if self.kwargs is None:
    #         self.kwargs = {}

@dataclass
class DatasetParameters:
    train: DatasetConfig
    validation: DatasetConfig
    # test: DatasetConfig | None = None

@dataclass
class EventAction:
    event: CallableEventWithFilter
    handler: Callable
    kwargs: dict | None = None # kwargs passed to handler (besides engine)

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

@dataclass
class ModelParameters:
    device: torch.device
    body: torch.nn.Module

@dataclass
class BrainNetParameters(ModelParameters):
    model = "BrainNet"
    heads: dict[str, torch.nn.Module]

@dataclass
class BrainRegParameters(ModelParameters):
    model = "BrainReg"
    svf: list[torch.nn.Module]

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
    load_from_dir: None | Path | str = None
    checkpoint_prefix: str = "state"
    checkpoint_subdir: str = "checkpoint"
    examples_subdir: str = "examples"
    evaluation_subdir: str = "evaluation"
    # Which images/surfaces to write when writing examples
    # The image used for prediction is called "x". None -> write everything
    examples_keys: list[str] | None = None
    save_checkpoint_on: CallableEventWithFilter = Events.EPOCH_COMPLETED(every=50)
    save_example_on: CallableEventWithFilter = Events.EPOCH_COMPLETED(every=20)
    checkpoint_filename_pattern: str = "{filename_prefix}_{name}_{global_step:05d}.pt"
    require_empty: bool = False # require that checkpoints do not already exist

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.load_from_dir = self.out_dir if self.load_from_dir is None else Path(self.load_from_dir)
        self._from_checkpoint_dir = self.load_from_dir / self.checkpoint_subdir
        self.checkpoint_dir = self.out_dir / self.checkpoint_subdir
        self.examples_dir = self.out_dir / self.examples_subdir
        self.evaluation_dir = self.out_dir / self.evaluation_subdir

        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)
        if not self.examples_dir.exists():
            self.examples_dir.mkdir(parents=True)
        if not self.evaluation_dir.exists():
            self.evaluation_dir.mkdir(parents=True)

@dataclass
class SynthesizerParameters:
    train: None | SynthesizerConfig = None
    validation: None | SynthesizerConfig = None
    test: None | SynthesizerConfig = None


@dataclass
class TrainParameters:
    max_epochs: int
    load_checkpoint: int = 0 # do not load
    epoch_length_train: int | None = 100
    epoch_length_val: int | None = 50
    gradient_accumulation_steps: int = 1
    evaluate_on: CallableEventWithFilter = Events.EPOCH_COMPLETED(every=10)
    events_trainer: list[EventAction] | None = None
    events_evaluators: list[EventAction] | None = None

    enable_amp: bool = True

    # Estimate inter-task affinity (ITA)
    enable_ITA_estimation: bool = False

    # Consider the model "converged" when the LR is reduced to this value
    minimum_lr: float = 1e-10

    def __post_init__(self):
        if self.events_trainer is None:
            self.events_trainer = []
        if self.events_evaluators is None:
            self.events_evaluators = []

@dataclass
class WandbParameters:
    # Logging using Weights & Biases (https://wandb.ai)
    enable: bool
    project: str
    name: str
    wandb_dir: Path | str
    log_on: CallableEventWithFilter
    run_id: None | str = None
    resume: None | str = "allow"
    tags: None | Sequence = None
    #fork_from: None | str = None

    # resume: str = "auto"

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
    model: ModelParameters | torch.nn.Module
    optimizer: OptimizerParameters
    results: ResultsParameters
    synthesizer: SynthesizerParameters | None
    train_params: TrainParameters
    wandb: WandbParameters
