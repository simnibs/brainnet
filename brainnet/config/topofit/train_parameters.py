from dataclasses import dataclass, InitVar

from brainsynth.config import DatasetConfig, SynthesizerConfig

import brainnet.config.train_parameters
import brainnet.networks


@dataclass(kw_only=True)
class TrainParameters(brainnet.config.train_parameters.TrainParameters):
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """

    project: str = "TopoFit"
    fov_out_size: list | tuple = (176, 208, 176)  # 1.00 isotropic (whole brain)
    # fov_out_size: list | tuple = (128, 208, 176) # 1.00 isotropic (single hemi)
    # fov_out_size: list | tuple = (224, 288, 224) # 0.75 isotropic (whole brain)
    # fov_out_size: list | tuple = (128, 288, 224) # 0.75 isotropic (single hemi)
    fov_out_center_str: str = "brain"
    package: InitVar[str] = __package__
    network: str = "TopoFit"

    # =====================================================================
    #   MODEL
    # =====================================================================

    # fmt: off
    UNET_ENCODER_CHANNELS   : InitVar[dict]         = {
        ("t1w", "050mm"):       [[16], [32], [64], [128], [256]], # [512]
        ("t1w", "075mm"):       [[16], [32], [64], [128], [256]],
        ("t1w", "1mm"):         [[16], [32], [64], [128], [256]],
        ("synth", "050mm"):     [[16], [32], [64], [128], [256]],
        ("synth", "075mm"):     [[16], [32], [64], [128], [256]],
        ("synth", "1mm"):       [[16], [32], [64], [128], [256]],
        ("synth", "random"):    [[16], [32], [64], [128], [256]],
    }
    UNET_DECODER_CHANNELS   : InitVar[dict]         = {
        ("t1w", "050mm"):       [[128], [64], [32], [16]],
        ("t1w", "075mm"):       [[128], [64], [32], [16]],
        ("t1w", "1mm"):         [[128], [64], [32], [16]],
        ("synth", "050mm"):     [[128], [96], [64], [32]], # no post conv
        ("synth", "075mm"):     [[128], [96], [64], [32]], # no post conv
        ("synth", "1mm"):       [[128], [96], [64], [32]],
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

    iterative_hemisphere_prediction: bool = False

    # =========================================================================
    #   PRETRAINED
    # =========================================================================

    # "/mnt/scratch/personal/jesperdn/results/TopoFit-Features/synth-1mm/checkpoint/state_checkpoint_00400.pt"
    load_body_from_checkpoint: str | None = None
    # "/mnt/scratch/personal/jesperdn/results/TopoFit/t1w_1mm-taubin_16/checkpoint/state_checkpoint_00600.pt"
    load_head_from_checkpoint: str | None = None

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

        surfaces = [
            dict(types="template", resolution=TOPOFIT_ORDER_IN),
            dict(types="white", resolution=TOPOFIT_ORDER_OUT, name="resample"),
            dict(types="pial", resolution=TOPOFIT_ORDER_OUT, name="resample"),
            # dict(types="sphere.reg", resolution=TOPOFIT_ORDER_OUT, name="resample"),
        ]

        self.dataset = {
            k: DatasetConfig(**v, surfaces=surfaces)
            for k, v in self.dataset_kwargs.items()
        }

        # =====================================================================
        # MODEL
        # =====================================================================

        UNET_DECODER_CHANNELS_POST = {
            ("t1w", "050mm"): None,
            ("t1w", "075mm"): None,
            ("t1w", "1mm"): None,
            ("synth", "050mm"): None,  # UNET_DECODER_CHANNELS["t1w", "050mm"],
            ("synth", "075mm"): None,  # UNET_DECODER_CHANNELS["t1w", "075mm"],
            ("synth", "1mm"): UNET_DECODER_CHANNELS["t1w", "1mm"],
            ("synth", "random"): UNET_DECODER_CHANNELS["t1w", "1mm"],
        }

        unet_kwargs = dict(
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
        graph_kwargs = dict(
            # in_channels=unet.num_features,
            in_order=TOPOFIT_ORDER_IN,
            out_order=TOPOFIT_ORDER_OUT,
            max_order=TOPOFIT_ORDER_MAX,
            # white_feature_maps=[unet.decoder_features]
            # * (TOPOFIT_ORDER_MAX - TOPOFIT_ORDER_IN + 1),
            white_channels=TOPOFIT_WHITE_MATTER_CHANNELS,
            # pial_feature_maps=unet.decoder_features,
            pial_channels=TOPOFIT_GRAY_MATTER_CHANNELS,
            pial_deform_module=TOPOFIT_GRAY_MATTER_MODULE,
        )
        if self.model_kwargs is None:
            self.model_kwargs = {}
        self.model = getattr(brainnet.networks, self.network)(
            unet_kwargs, graph_kwargs, **self.model_kwargs
        )

        # =====================================================================
        # SYNTHESIZER
        # =====================================================================

        match self.resolution:
            case "050mm":
                in_res = (0.5, 0.5, 0.5)
                builder_res = "Iso"
            case "075mm":
                in_res = (0.75, 0.75, 0.75)
                builder_res = "Iso"
            case "1mm":
                in_res = (1.0, 1.0, 1.0)
                builder_res = "Iso"
            case "random":
                in_res = (1.0, 1.0, 1.0)
                builder_res = ""
            case _:
                raise ValueError(
                    f"Invalid resolution specification ({self.resolution})"
                )
        builder_contrast = "Synth" if self.contrast == "synth" else "Select"

        if self.builder_train is None:
            self.builder_train = f"Only{builder_contrast}{builder_res}"
        if self.builder_validation is None:
            if self.validation_image == "native":
                self.builder_validation = f"OnlySelect{builder_res}"
            elif self.validation_image == "synth":
                self.builder_validation = f"OnlySynth{builder_res}"
            else:
                raise ValueError(f"Invalid 'validation_image' {self.validation_image}")

        self.synthesizer = dict(
            train=SynthesizerConfig(
                self.builder_train,
                in_res,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
                # segmentation_labels = "brainseg"
                selectable_images=self.selectable_images_train,
                device=self.device,
                **self.preprocessor_train_kwargs,
            ),
            validation=SynthesizerConfig(
                self.builder_validation,
                in_res,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
                # segmentation_labels = "brainseg"
                selectable_images=self.selectable_images_validation,
                device=self.device,
                **self.preprocessor_validation_kwargs,
            ),
        )

        if self.network == "TopoFitSingleHemi":
            # Must be true for this network
            self.iterative_hemisphere_prediction = True

        # Store information needed for setting up the prediction
        self.prediction_config = dict(
            step=dict(
                iterative_hemisphere_prediction=self.iterative_hemisphere_prediction
            ),
            preprocessor=dict(
                in_res=in_res,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
            ),
            model=dict(
                topofit=graph_kwargs,
                unet=unet_kwargs,
            ),
        )
