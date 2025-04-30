from dataclasses import dataclass

from brainsynth.config import DatasetConfig, SynthesizerConfig
from brainsynth.dataset import AlignmentDataset

import brainnet.config.train_parameters
from brainnet.modules.alignment import AffineCorticalAlignment


@dataclass(kw_only=True)
class TrainParameters(brainnet.config.train_parameters.TrainParameters):
    """Training parameters (and default values) are defined here.

    Additional setup is performed in post_init as inferred from these variables.

    """

    fov_out_size = (192, 224, 192)
    fov_out_center_str = "image"

    package = __package__

    def __post_init__(self, *args):
        super().__post_init__(*args)

        self.dataloader = dict(dataset_class=AlignmentDataset)

        # =====================================================================
        # DATASET
        # =====================================================================

        self.dataset = dict(
            train=DatasetConfig(
                **self.dataset_kwargs["train"],
                target_vertices=None,
                template_surface=None,
            ),
            validation=DatasetConfig(
                **self.dataset_kwargs["validation"],
                target_vertices=None,
                template_surface=None,
            ),
        )

        # =====================================================================
        # MODEL
        # =====================================================================

        self.model = AffineCorticalAlignment(device=self.device)

        # =====================================================================
        # SYNTHESIZER
        # =====================================================================
        # Builder configuration
        match self.contrast, self.resolution:
            case "t1w", "1mm":
                builder_train = "CropSelectIso"
                builder_validation = "CropSelectIso"
            case "synth", "1mm":
                builder_train = "CropSynthIso"
                builder_validation = "CropSelectIso"
            case "synth", "random":
                builder_train = "CropSynth"
                builder_validation = "CropSelect"

        img_sel_train = None if self.contrast == "synth" else self.contrast
        img_sel_val = "t1w" if self.contrast == "synth" else self.contrast

        self.synthesizer = dict(
            train=SynthesizerConfig(
                builder=builder_train,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
                # segmentation_labels = "brainseg"
                # photo_mode = False
                selectable_images=img_sel_train,
                device=self.device,
            ),
            validation=SynthesizerConfig(
                builder=builder_validation,
                out_size=self.fov_out_size,
                out_center_str=self.fov_out_center_str,
                # segmentation_labels = "brainseg"
                # photo_mode = False
                selectable_images=img_sel_val,  # ["t1w", "t2w", "flair"],
                device=self.device,
            ),
        )
