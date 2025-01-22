import copy
import functools

import torch
from torch.utils.data import default_collate
from brainnet.modules.head import surface_modules

# import nibabel as nib
# import torch

# from brainsynth.dataset import GenericDataset

# from brainnet.resources import PretrainedModels
from brainsynth.utilities import apply_affine
from brainnet.utilities import recursively_apply_function

# from pathlib import Path


class PredictionStep:
    def __init__(self, preprocessor, model, enable_amp: bool = False):
        self.preprocessor = preprocessor
        self.model = model
        self.enable_amp = enable_amp

        module = [i for i in self.model.heads.values() if isinstance(i, surface_modules)]
        assert len(module) == 1
        module = module[0]

        topology = module.out_topology
        self.topology = dict(lh=topology, rh=copy.deepcopy(topology))
        self.topology["rh"].reverse_face_orientation()

    def prepare_batch(self, batch):
        # decollate and loop?
        image, template, vox_to_mri = batch
        batch = self.preprocessor(
            images=dict(image=image),
            surfaces={},
            initial_vertices=template,
            affines=dict(image=vox_to_mri)
        )
        return tuple(default_collate([b]) for b in batch)

    def __call__(self, engine, batch):
        images, _, template, vox_to_mri = self.prepare_batch(batch)
        image = images["image"]

        self.model.eval()
        with torch.inference_mode():
            if self.model.device.type == "cpu":
                y_pred = self.model(image, template)
            else:
                with torch.autocast(self.model.device.type, enabled=self.enable_amp):
                    y_pred = self.model(image, template)

            # remove batch dimension
            func = functools.partial(torch.squeeze, dim=0),
            y_pred = recursively_apply_function(y_pred, func)

            # convert from voxel to MRI space
            if "surface" in y_pred:
                func = functools.partial(apply_affine, vox_to_mri),
                y_pred["surface"] = recursively_apply_function(y_pred["surface"], func)

        return y_pred, vox_to_mri


# device = torch.device("cuda:0")

# # Dataset
# dataset = GenericDataset(
#     images=["/home/jesperdn/nobackup/eugenio/466137_conform.nii.gz"],
#     mni_transform=["/home/jesperdn/nobackup/eugenio/affine.txt"],
# )

# name = "topofit"
# specs = ("synth", "random")
# pretrained_models = PretrainedModels()
# model = pretrained_models.load_model(name, specs, device)
# preprocessor = pretrained_models.load_preprocessor(name, specs, device)

# predict_step = PredictionStep(preprocessor, model, enable_amp=True)


# vol_info = dict(
#     head=[2, 0, 20],
#     valid="1  # volume info valid",
#     filename="vol.nii",
#     voxelsize=[1, 1, 1],
#     volume=(0, 0, 0),
#     xras=[-1, 0, 0],
#     yras=[0, 0, -1],
#     zras=[0, 1, 0],
#     cras=[0, 0, 0],
# )
# vol_info["volume"] = (256, 256, 256)

# predictions = []
# for batch in dataset:
#     y_pred, vox_to_mri = predict_step(None, batch)
#     y_pred = y_pred["surface"]

#     # convert from voxel to MRI space
#     for func in [
#         functools.partial(apply_affine, vox_to_mri),
#         functools.partial(torch.squeeze, dim=0),
#     ]:
#         y_pred = recursively_apply_function(y_pred, func)

#     out_dir = Path("/home/jesperdn/nobackup/eugenio/")
#     # if not out_dir.exists():
#     #     out_dir.mkdir(parents=True)

#     for hemi, s in y_pred.items():
#         for surf, ss in s.items():
#             v = ss
#             nib.freesurfer.write_geometry(
#                 out_dir / f"{hemi}.{surf}",
#                 v.cpu().numpy(),
#                 predict_step.topology[hemi].faces.cpu().numpy(),
#                 volume_info=vol_info
#             )
