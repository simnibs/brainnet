from dataclasses import dataclass, InitVar

from brainsynth.config import DatasetConfig, SynthesizerConfig
from brainsynth.dataset import AlignmentDataset

import brainnet.config.train_parameters
from brainnet.networks.templatereg import TemplateRegAffine


@dataclass(kw_only=True)
class TrainParameters(brainnet.config.train_parameters.TrainParameters):
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """

    project: str = "TemplateRegAffine"
    fov_out_size: list | tuple = (192, 224, 192)
    fov_out_center_str: str = "image"
    package: InitVar[str] = __package__

    def __post_init__(self, *args):
        super().__post_init__(*args)

        self.dataloader = dict(dataset_class=AlignmentDataset)

        # =====================================================================
        # DATASET
        # =====================================================================
        self.dataset = dict(
            train=DatasetConfig(**self.dataset_kwargs["train"], surfaces=None),
            validation=DatasetConfig(
                **self.dataset_kwargs["validation"], surfaces=None
            ),
        )

        # =====================================================================
        # MODEL
        # =====================================================================

        model = dict(
            spatial_dims=3,
            pre_conv_ch=8,
            encoder_channels=[[32], [32], [32], [32], [32]],
            # encoder_channels=[[16], [16], [32], [32], [64]],
            decoder_channels=[[32], [32], [32], [32]],
            # decoder_channels=[[32], [32], [16], [32]],
            points_per_hemisphere=16,
            weigh_by_feature_mass=True,
        )

        self.model = TemplateRegAffine(**model)

        # =====================================================================
        # SYNTHESIZER
        # =====================================================================

        if self.builder_train is None:
            match self.contrast, self.resolution:
                case "t1w", "1mm":
                    self.builder_train = "CropSelectIso"
                case "synth", "1mm":
                    self.builder_train = "CropSynthIso"
                case "synth", "random":
                    self.builder_train = "CropSynth"
        if self.builder_validation is None:
            match self.contrast, self.resolution:
                case "t1w", "1mm":
                    self.builder_validation = "CropSelectIso"
                case "synth", "1mm":
                    self.builder_validation = "CropSelectIso"
                case "synth", "random":
                    self.builder_validation = "CropSelect"

        self.synthesizer = dict(
            train=SynthesizerConfig(
                builder=self.builder_train,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
                # segmentation_labels = "brainseg"
                # photo_mode = False
                selectable_images=self.selectable_images_train,
                device=self.device,
                **self.builder_train_kw,
            ),
            validation=SynthesizerConfig(
                builder=self.builder_validation,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
                # segmentation_labels = "brainseg"
                # photo_mode = False
                selectable_images=self.selectable_images_validation,
                device=self.device,
                **self.builder_validation_kw,
            ),
        )

        self.prediction_config = dict(
            preprocessor=dict(
                out_size=self.fov_out_size, out_center_str=self.fov_out_center_str
            ),
            model=model,
        )
