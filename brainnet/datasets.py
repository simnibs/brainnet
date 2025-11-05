from pathlib import Path
from typing import Callable

import nibabel as nib
import nibabel.processing
import numpy as np
import torch

from brainsynth.dataset import _load_image
from brainsynth.utilities import apply_affine

from brainnet.mesh.surface import load_deepsurfer_template

# (8) from https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
MNI305_to_MNI152 = torch.tensor(
    [
        [0.9975, -0.0073, 0.0176, -0.0429],
        [0.0146, 1.00090003, -0.0024, 1.54960001],
        [-0.013, -0.0093, 0.9971, 1.18400002],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def reorient_to_ras(img):
    """Reorient image (and possibly flip dimensions) so as to bring it as close
    as possible to having an identity affine transformation matrix.
    """
    perm, flip = (
        nib.orientations.io_orientation(np.linalg.inv(img.affine)).astype(int).T
    )

    # Construct new affine
    affine = np.identity(4)
    affine[:3, :3] = img.affine[:3, perm] * flip
    affine[:3, 3] = img.affine[:3, 3]

    # Adjust image data accordingly
    data = img.get_fdata().transpose(perm)
    shape = data.shape
    for i, f in enumerate(flip):
        if f == -1:
            affine[:3, 3] -= affine[:3, i] * (shape[i] - 1)
            data = np.ascontiguousarray(np.flip(data, i))

    dtype = img.get_data_dtype()

    return nib.Nifti1Image(data.astype(dtype), affine)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images: list[Path] | list[str] | tuple, conform: bool = False):
        """Dataset formed from a list of images.

        Images are loaded using `nibabel.load`

        Parameters
        ----------
        images :
            List of filenames of images in the dataset.
        conform: bool
            If the linear part of the affine matrix is not equal to identity,
            resample the image such that it is (uses
            nibabel.processing.conform). If false, the image is assumed to be
            correctly preprocessed.

        """
        self.images = images
        self.conform = conform

    def preprocess_image(self, img):
        # Apply conform if the linear part of the affine deviates identity,
        # i.e., we want 1 mm voxels aligned the major axes in RAS orientation.
        if self.conform and not np.allclose(img.affine[:3, :3], np.identity(3)):
            img = nibabel.processing.conform(nib.funcs.squeeze_image(img))
        return img

    def load_image(self, index):
        return _load_image(
            self.images[index],
            torch.float,
            image_transform=self.preprocess_image,
            return_affine=True,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, vox2mri = self.load_image(index)
        return image, vox2mri


class TopoFitDataset(ImageDataset):
    def __init__(
        self,
        images: list | tuple[Path, str],
        conform: bool = False,
        mni_transforms: list | tuple[Path, str] | None = None,
        mni_direction: str = "mni2sub",
        mni_space: str = "mni152",
        hemi: str = "both",
        template_order: int = 0,
        mni_transform_loader: Callable = np.loadtxt,
    ):
        """Dataset formed from a list of images and transformations.


        Parameters
        ----------

        conform: bool
            If the linear part of the affine matrix is not equal to identity,
            resample the image such that it is (uses
            nibabel.processing.conform). If false, the image is assumed to be
            correctly preprocessed.

        """
        super().__init__(images, conform)

        if mni_transforms is not None:
            assert len(images) == len(mni_transforms)
        assert mni_space in ("mni152", "mni305")
        assert mni_direction in {"mni2sub", "sub2mni"}
        assert hemi in ("both", "lh", "rh")
        self.hemi = ("lh", "rh") if hemi == "both" else [hemi]

        self.mni_transform = mni_transforms
        self.mni_space = mni_space
        self.mni_direction = mni_direction
        self.mni_transform_loader = mni_transform_loader

        self.template = {
            k: v.vertices.squeeze(0)
            for k, v in load_deepsurfer_template(
                template_order, "white", self.hemi
            ).items()
        }

        match mni_direction:
            case "sub2mni":

                def mni_to_sub(t):
                    return torch.linalg.inv(t)
            case "mni2sub":

                def mni_to_sub(t):
                    return t

        match self.mni_space:
            case "mni152":
                # the template vertices are in mni305 so add a transformation
                # from mni305 to mni152
                def preprocess_mni_transform(t):
                    return mni_to_sub(t) @ MNI305_to_MNI152
            case "mni305":

                def preprocess_mni_transform(t):
                    return mni_to_sub(t)

        self.preprocess_mni_transform = preprocess_mni_transform

    def prepare_template(self, trans, vox2mri):
        trans = torch.linalg.inv(vox2mri) @ trans
        return {k: apply_affine(trans, v) for k, v in self.template.items()}

    def get_template(self, index, vox2mri):
        if self.mni_transform is None:
            template = None
        else:
            trans = self.mni_transform_loader(self.mni_transform[index])
            trans = torch.tensor(trans, dtype=torch.float)
            trans = self.preprocess_mni_transform(trans)
            template = self.prepare_template(trans, vox2mri)
        return template

    def __getitem__(self, index):
        image, vox2mri = super().__getitem__(index)
        template = self.get_template(index, vox2mri)
        return image, vox2mri, template
