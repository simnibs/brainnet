from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class DatasetConfig:
    train: dict
    validation: dict

@dataclass
class ModelConfig:
    """Filename is only used if config is not specified."""
    device: str
    body: dict
    heads: dict

class OptimizerConfig:
    name: str
    kwargs: dict
    lr_parameter_groups: dict | None = None


optconf = OptimizerConfig("AdamW", dict(lr=1.0e-4))

    # lr_factor: 0.5 # applied after optimizer state is loaded

    # body: BrainNet.body.[]
    # heads:  BrainNet.heads.[]
    # lr_parameter_groups:
        # body: 1.0e-4
        # heads:
        # surface: 1.0e-4
        # segmentation: 1.0e-3


@dataclass
class ResultsConfig:
    out_dir: Path | str

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)


@dataclass
class WandbConfig:
    enable: bool
    project: str
    name: str
    wandb_dir: str
    resume: str = "auto"
    kwargs: dict | None = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

@dataclass
class DataloaderConfig:
    batch_size: int = 1
    num_workers: int = 4
    prefetch_factor: int = 2
    kwargs: dict | None = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

@dataclass
class LossConfig:
    functions: dict
    head_weights: dict
    loss_weights: dict



# Logging using Weights & Biases (https://wandb.ai)
wandb = dict(
    enable = True,
    project = project,
    name = run_name,      # name used to identify run in UI
    resume = True,
    # tags =
    # entity = null  # username/team name to send data to
    # dir = null      # sub dir of results dir
)





@dataclass
class EventAction:
    event: str
    event_filter: dict
    action: Callable
    action_kwargs: dict | None



loss_events = [
    EventAction(
        event="EPOCH_COMPLETED",
        event_filter=dict(once=200),
        action=set_loss_weight,
        action_kwargs={("white", "hinge"): 10.0},
    ),
    EventAction(
        event="EPOCH_COMPLETED",
        event_filter=dict(once=600),
        action=set_loss_weight,
        action_kwargs={
            ("white", "matched"):   0.01, # /= 100
            ("white", "hinge"):     0.1, # /= 100
            ("white", "chamfer"):   1.0, # new
            ("white", "curv"):     50.0, # new

            ("pial", "matched"):    0.01,
            ("pial", "hinge"):      0.1,
            ("pial", "chamfer"):    1.0, # new
            ("pial", "curv"):      25.0, # new; white / 2
        },
    ),
    EventAction(
        event="EPOCH_COMPLETED",
        event_filter=dict(once=1000),
        action=set_loss_weight,
        action_kwargs={
            ("white", "matched"):   0.001,
            ("white", "hinge"):     0.01,
            ("white", "curv"):     10.0,

            ("pial", "matched"):    0.001,
            ("pial", "hinge"):      0.01,
            ("pial", "curv"):      10.0,
        },
    ),
    EventAction(
        event="EPOCH_COMPLETED",
        event_filter=dict(once=1400),
        action=set_loss_weight,
        action_kwargs={
            ("white", "matched"):   0.0,
            ("white", "hinge"):     0.0,
            ("white","edge"):      2.5,

            ("pial", "matched"):    0.0,
            ("pial", "hinge"):      0.0,
            ("pial","edge"):       2.5,
            ("pial", "curv"):       5.0, # white / 2
        },
    ),
    EventAction(
        event="EPOCH_COMPLETED",
        event_filter=dict(once=1800),
        action=set_loss_weight,
        action_kwargs={
            ("white","edge"):      1.0,

            ("pial","edge"):       1.0,
            ("pial", "curv"):       2.5, # white / 4
        },
    ),
]


for e in loss_events:
    engine.add_event_handler(
        getattr(Events, e.event)(**e.event_filter),
        e.action(**e.action_kwargs)
    )





@dataclass
class TrainConfig:
    project: str
    run: str
    device: str

    results: ResultsConfig
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    synthesizer: dict
    model: ModelConfig
    loss: LossConfig

    events: dict
    wandb: WandbConfig

class TrainConfig:

    project_name = "BrainNet"
    run_name = "LH_run1"

    device = "cuda:0"
    target_surface_resolution = 5
    target_surface_hemisphere = "lh"
    initial_surface_resolution = 0

    # results are stored in results_dir/PROJECT_NAME/RUN_NAME/
    self.results = dict(
        dir = "/mnt/scratch/personal/jesperdn/results" # PROJECT_NAME will be appended

    )


    # This section defines keyword arguments passed to DatasetConfig

    dataset = DatasetConfig(
        train=dict(
            root_dir = "/mnt/projects/CORTECH/nobackup/training_data",
            subject_dir = "/mnt/projects/CORTECH/nobackup/training_data_subjects",
            subject_subset = "train",
            # synthesizer = None,
            # datasets = ["ABIDE", "HCP"], # default: all
            # images = ["generation_labels", "t1w"],
            # ds_structure = "flat",
            target_surface_resolution = target_surface_resolution,
            target_surface_hemispheres = target_surface_hemisphere,
            initial_surface_resolution = initial_surface_resolution,
        ),
        validation=dict(
            root_dir = "/mnt/projects/CORTECH/nobackup/training_data",
            subject_dir = "/mnt/projects/CORTECH/nobackup/training_data_subjects",
            subject_subset = "validation",
            target_surface_resolution = target_surface_resolution,
            target_surface_hemispheres = target_surface_hemisphere,
            initial_surface_resolution = initial_surface_resolution,
        ),
    )

    dataloader = dict(
        batch_size = 1,
        num_workers = 4,
        prefetch_factor = 2, # this seems to be PER dataset..?
    )


    # kwargs to SynthesizerConfig
    synthesizer = dict(
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
    )

    model = ModelConfig(
        device = device,
        # > contents of this node is passed to BrainNet
        # model. !include models/foundation.yaml
        body = dict(
            model = "UNet",
            kwargs = dict(
                spatial_dims = 3,
                in_channels = 1,
                encoder_channels = [[32], [64], [96], [128], [160]],
                decoder_channels = [[128], [96], [64], [64]],
            )
        ),
        heads = dict(
            surface = dict(
                module = dict(
                    name = "SurfaceModule",
                    kwargs = dict(prediction_res = target_surface_resolution),
                ),
                runtime_kwargs = dict(return_pial = True),
            ),
        ),
    )



    [loss]
    filename = "path/to/model.model"

    # Logging using Weights & Biases (https://wandb.ai)
    wandb = dict(
        enable = True,
        project_name = project_name,
        name = run_name,      # name used to identify run in UI
        resume = True,
        # tags =
        # entity = null  # username/team name to send data to
        # dir = null      # sub dir of results dir

    )

