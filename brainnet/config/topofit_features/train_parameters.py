from dataclasses import dataclass, InitVar

from brainsynth.config import DatasetConfig

from brainnet.config.topofit import train_parameters
import brainnet.networks


@dataclass(kw_only=True)
class TrainParameters(train_parameters.TrainParameters):
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """

    # =========================================================================
    #   PRETRAINED
    # =========================================================================
    pretrained_run: str  # the pretrained run defining the target features
    pretrained_checkpoint: int  # checkpoint of the pretrained run
    pretrained_project: str = "TopoFit"

    project: str = "TopoFit-Features"

    package: InitVar[str] = __package__
    network = "TopoFit"

    def __post_init__(self, *args):
        # split post init args for parent class and child class
        (
            *super_args,
            UNET_ENCODER_CHANNELS,
            UNET_DECODER_CHANNELS,
            UNET_RETURN_ENCODER_FEATURES,
            UNET_RETURN_DECODER_FEATURES,
            TOPOFIT_ORDER_IN,
            TOPOFIT_ORDER_OUT,
            TOPOFIT_ORDER_MAX,
            TOPOFIT_WHITE_MATTER_CHANNELS,
            TOPOFIT_GRAY_MATTER_CHANNELS,
            TOPOFIT_GRAY_MATTER_MODULE,
        ) = args
        super().__post_init__(*args)

        self.pretrained_checkpoint_filename = (
            self.results_dir
            / self.pretrained_project
            / self.pretrained_run
            / "checkpoint"
            / f"state_checkpoint_{self.pretrained_checkpoint:05d}.pt"
        )

        # =====================================================================
        # DATASET
        # =====================================================================

        # self.dataset_kwargs["train"]["images"] += ["brain_dist_map", "t1w"]
        # self.dataset_kwargs["validation"]["images"] += ["brain_dist_map"]

        surfaces = [
            dict(types="template", resolution=TOPOFIT_ORDER_IN),
        ]

        self.dataset = dict(
            train=DatasetConfig(
                **self.dataset_kwargs["train"],
                surfaces=surfaces,
            ),
            validation=DatasetConfig(
                **self.dataset_kwargs["validation"],
                surfaces=surfaces,
            ),
        )

        # =====================================================================
        # PRETRAINED MODEL
        # =====================================================================

        unet_kwargs = dict(
            spatial_dims=3,
            in_channels=1,
            encoder_channels=UNET_ENCODER_CHANNELS["t1w", "1mm"],
            decoder_channels=UNET_DECODER_CHANNELS["t1w", "1mm"],
            return_encoder_features=UNET_RETURN_ENCODER_FEATURES,
            return_decoder_features=UNET_RETURN_DECODER_FEATURES,
            # match the synth features to the T1w features
            encoder_post=None,
            decoder_post=None,
        )

        self.pretrained_model = getattr(brainnet.networks, self.network)(
            unet_kwargs, self.prediction_config["model"]["topofit"]
        )
        self.pretrained_dir = self.results.out_dir
