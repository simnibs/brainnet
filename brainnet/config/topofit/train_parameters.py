from dataclasses import dataclass, InitVar

from brainsynth.config import DatasetConfig, SynthesizerConfig

from brainnet import config
import brainnet.config.train_parameters
from brainnet.modules import body, head


@dataclass(kw_only=True)
class TrainParameters(brainnet.config.train_parameters.TrainParameters):
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """

    project: str = "TopoFit"
    fov_out_size: list | tuple = (176, 208, 176)
    fov_out_center_str: str = "brain"
    package: InitVar[str] = __package__

    # =====================================================================
    #   MODEL
    # =====================================================================

    # fmt: off
    UNET_ENCODER_CHANNELS   : InitVar[dict]         = {
        ("t1w", "1mm"):         [[16], [32], [64], [128], [256]],
        ("synth", "1mm"):       [[16], [32], [64], [128], [256]],
        ("synth", "random"):    [[16], [32], [64], [128], [256]],
    }
    UNET_DECODER_CHANNELS   : InitVar[dict]         = {
        ("t1w", "1mm"):         [[128], [64], [32], [16]],
        ("synth", "1mm"):       [[128], [96], [64], [64]],
        ("synth", "random"):    [[128], [96], [64], [64]],
    }
    UNET_RETURN_ENCODER_FEATURES: InitVar[list | None] = None
    UNET_RETURN_DECODER_FEATURES: InitVar[list | None] = [True, True, True, True]

    UNET_FREEZE: bool = False

    TOPOFIT_ORDER_IN        : InitVar[int]          = 0
    TOPOFIT_ORDER_OUT       : InitVar[int]          = 6
    TOPOFIT_ORDER_MAX       : InitVar[int]          = 6
    TOPOFIT_WHITE_MATTER_CHANNELS: InitVar[dict] = dict(
        encoder=[96, 96, 96, 96],
        decoder=[96, 96, 96],
    )
    TOPOFIT_GRAY_MATTER_CHANNELS: InitVar[list]     = [32]
    TOPOFIT_GRAY_MATTER_MODULE: InitVar[str]        = "LinearDeformationBlock"
    # fmt: on

    # =========================================================================
    #   PRETRAINED
    # =========================================================================

    load_body_from_checkpoint: str | None = (
        "/mnt/scratch/personal/jesperdn/results/TopoFit-UNet/synth_1mm/checkpoint/state_checkpoint_00400.pt"
    )
    load_head_from_checkpoint: str | None = (
        None  # "/mnt/scratch/personal/jesperdn/results/TopoFit/t1w_1mm-taubin_16/checkpoint/state_checkpoint_00600.pt"
    )

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
        super().__post_init__(*super_args)

        self.dataloader = dict()

        # =====================================================================
        # DATASET
        # =====================================================================

        template_surface = dict(resolution=TOPOFIT_ORDER_IN, name="template")
        target_vertices = dict(resolution=TOPOFIT_ORDER_OUT, name="target")

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
        # MODEL
        # =====================================================================

        UNET_DECODER_CHANNELS_POST = {
            ("t1w", "1mm"): None,
            ("synth", "1mm"): UNET_DECODER_CHANNELS["t1w", "1mm"],
            ("synth", "random"): UNET_DECODER_CHANNELS["t1w", "1mm"],
        }

        unet = body.UNet(
            spatial_dims=3,
            in_channels=1,
            encoder_channels=UNET_ENCODER_CHANNELS[self.contrast, self.resolution],
            decoder_channels=UNET_DECODER_CHANNELS[self.contrast, self.resolution],
            return_encoder_features=UNET_RETURN_ENCODER_FEATURES,
            return_decoder_features=UNET_RETURN_DECODER_FEATURES,
            # match the synth features to the T1w features
            encoder_post=None,
            decoder_post=UNET_DECODER_CHANNELS_POST[self.contrast, self.resolution],
        )

        topofit = head.TopoFit(
            in_channels=unet.num_features,
            in_order=TOPOFIT_ORDER_IN,
            out_order=TOPOFIT_ORDER_OUT,
            max_order=TOPOFIT_ORDER_MAX,
            white_feature_maps=[unet.decoder_features]
            * (TOPOFIT_ORDER_MAX - TOPOFIT_ORDER_IN + 1),
            white_channels=TOPOFIT_WHITE_MATTER_CHANNELS,
            pial_feature_maps=unet.decoder_features,
            pial_channels=TOPOFIT_GRAY_MATTER_CHANNELS,
            pial_deform_module=TOPOFIT_GRAY_MATTER_MODULE,
            device=self.device,
        )

        self.model = config.BrainNetParameters(
            device=self.device,
            body=unet,
            heads=dict(surface=topofit),
        )

        # =====================================================================
        # SYNTHESIZER
        # =====================================================================

        match self.resolution:
            case "1mm":
                builder_res = "Iso"
            case "random":
                builder_res = ""
            case _:
                raise ValueError

        random_skullstrip: bool = True

        builder_contrast = "Synth" if self.contrast == "synth" else "Select"
        if builder_contrast == "Synth" or random_skullstrip:
            # synth has skullstrip anyway
            builder_train = f"Only{builder_contrast}{builder_res}"
        else:
            builder_train = f"Only{builder_contrast}NoSkullStrip{builder_res}"
        builder_validation = f"OnlySelectNoSkullStrip{builder_res}"

        img_sel_train = None if self.contrast == "synth" else [self.contrast]
        img_sel_val = ["t1w"] if self.contrast == "synth" else [self.contrast]

        self.synthesizer = dict(
            train=SynthesizerConfig(
                builder=builder_train,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
                # segmentation_labels = "brainseg"
                selectable_images=img_sel_train,
                device=self.device,
            ),
            validation=SynthesizerConfig(
                builder=builder_validation,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
                # segmentation_labels = "brainseg"
                selectable_images=img_sel_val,
                device=self.device,
            ),
        )
