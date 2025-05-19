import copy
from dataclasses import dataclass, InitVar

from brainsynth.config import DatasetConfig

from brainnet import config
from brainnet.config.topofit import train_parameters
from brainnet.modules import body


@dataclass(kw_only=True)
class TrainParameters(train_parameters.TrainParameters):
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """

    project: str = "TopoFit-UNet"

    package: InitVar[str] = __package__

    # =========================================================================
    #   PRETRAINED
    # =========================================================================

    load_pretrained_model_from_checkpoint = "/mnt/scratch/personal/jesperdn/results/TopoFit/t1w-1mm-neglogprob-std/checkpoint/state_checkpoint_00540.pt"

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

        # =====================================================================
        # DATASET
        # =====================================================================

        self.dataset_kwargs["train"]["images"] += ["brain_dist_map", "t1w"]
        self.dataset_kwargs["validation"]["images"] += ["brain_dist_map"]

        template_surface = dict(resolution=TOPOFIT_ORDER_IN, name="template")
        target_vertices = None

        self.dataset = dict(
            train=DatasetConfig(
                **self.dataset_kwargs["train"],
                target_vertices=target_vertices,
                template_surface=template_surface,
            ),
            validation=DatasetConfig(
                **self.dataset_kwargs["validation"],
                target_vertices=target_vertices,
                template_surface=template_surface,
            ),
        )

        # =====================================================================
        # PRETRAINED MODEL
        # =====================================================================

        pre_unet = body.UNet(
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

        self.pretrained_model = config.BrainNetParameters(
            device=self.device,
            body=pre_unet,
            heads=copy.deepcopy(self.model.heads),
        )
