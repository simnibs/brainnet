import argparse
import importlib
from pathlib import Path
import sys

import nibabel as nib

import torch
from torch.utils.data import default_collate
from ignite.engine import Engine

from brainsynth.constants import IMAGE

import brainnet.initializers
import brainnet.train.utilities
from brainnet.train.brainnet_train import SupervisedStep

from brainsynth.transforms import EnsureDevice, IntensityNormalization, PadTransform, SpatialCrop, SpatialCropParameters, AdjustAffineToSpatialCrop, TranslationTransform



class PredictionStep(SupervisedStep):
    def __init__(self, model, criterion, out_size, enable_amp: bool = False):
        super().__init__(None, model, criterion)
        self.enable_amp = enable_amp
        self.intensity_normalization = IntensityNormalization()
        self.ensure_device = EnsureDevice(model.device)
        self.out_size = torch.tensor(out_size)

    def get_brain_center(self, init_verts):

        bbox = torch.stack((init_verts["lh"].amin(0), init_verts["lh"].amax(0)))
        bbox[0] = torch.minimum(bbox[0], init_verts["rh"].amin(0))
        bbox[1] = torch.maximum(bbox[1], init_verts["rh"].amax(0))

        return bbox.mean(0)

    def crop_batch(self, batch, affine):
        images, surfaces, init_verts = batch

        first_image = next(iter(images.values()))
        spatial_size = first_image.shape[-3:]

        out_center = self.get_brain_center(init_verts)

        crop_params = SpatialCropParameters(self.out_size, out_center)(spatial_size)
        cropper = SpatialCrop(spatial_size, crop_params["slices"])
        padder = PadTransform(crop_params["pad"])
        adjuster = AdjustAffineToSpatialCrop(torch.tensor(crop_params["offset"], dtype=torch.float))

        affine_out = adjuster(affine)

        # Crop image and adjust affine accordingly
        for k,v in images.items():
            images[k] = padder(cropper(v))

        # surface translation
        # surface_translator = TranslationTransform(affine[:3,3]-affine_out[:3,3])
        surface_translator = TranslationTransform(crop_params["offset"], invert=True)
        for k,v in init_verts.items():
            init_verts[k] = surface_translator(v)
        for k,v in surfaces.items():
            for kk,vv in v.items():
                surfaces[k][kk] = surface_translator(vv)

        return images, surfaces, init_verts, affine_out

    def prepare_batch(self, batch, affine):
        batch = self.crop_batch(batch, affine)

        images, surfaces, init_verts, affine = default_collate([batch])
        images = self.ensure_device(images)
        surfaces = self.ensure_device(surfaces)
        init_verts = self.ensure_device(init_verts)

        image = self.intensity_normalization(images["t2w"])

        return image, surfaces, init_verts, affine

    def __call__(self, engine, batch, affine):
        image, y_true, init_verts, affine = self.prepare_batch(batch, affine)

        self.model.eval()
        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                y_pred = self.model(image, init_verts)

        return image, y_pred, affine, y_true


def predict(args):

    """

    train_setup_file = "brainnet.config.topofit.mri.main"
    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")

    args = parse_args(["blabla", "brainnet.config.topofit.mri.main", "1600", "/mnt/scratch/personal/jesperdn/topofit-ours"])

    """

    train_setup_file = args.config  # "brainnet.config.cortex.main"

    print("Setting up prediction...")

    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")
    train_setup.train_params.load_checkpoint = args.checkpoint

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    criterion = brainnet.initializers.init_criterion(train_setup.criterion)[args.subset]
    dataloader = brainnet.initializers.init_dataloader(
        train_setup.dataset, train_setup.dataloader
    )[args.subset]
    model = brainnet.initializers.init_model(train_setup.model)
    # synth = brainnet.initializers.init_synthesizer(train_setup.synthesizer)[args.subset]

    eval_step = PredictionStep(
        model,
        criterion,
        enable_amp=train_setup.train_params.enable_amp,
        out_size=[176, 208, 176],
    )
    #evaluator = Engine(eval_step)

    to_load = dict(
        model=model,
        **{f"criterion[{args.subset}]": criterion},
    )
    brainnet.train.utilities.load_checkpoint(to_load, train_setup)

    print(f"Setup completed. Predicting at epoch {args.checkpoint}")

    sep_line = 79 * "="

    print(sep_line)
    print(f"Config file     {train_setup_file}")
    print(f"Project         {train_setup.project:30s}")
    print(f"Run             {train_setup.run:30s}")
    print("Prediction settings")
    print(f"  Output dir    {out_dir}")
    print(f"  Subset        {args.subset}")
    print(sep_line)

    faces = eval_step.surface_template["y_pred"]["lh"]["white"].topology.faces.cpu().numpy()

    vol_info = dict(
        head=[2, 0, 20],
        valid="1  # volume info valid",
        filename="vol.nii",
        voxelsize=[1, 1, 1],
        volume=(0, 0, 0),
        xras=[-1, 0, 0],
        yras=[0, 0, -1],
        zras=[0, 1, 0],
        cras=[0, 0, 0],
    )

    for dataset in dataloader.dataset.datasets:
        print(dataset)
        first_img = dataset.images[0]
        img = getattr(IMAGE.images, first_img)
        ds_dir = dataset.ds_dir / dataset.name
        for i,sub in enumerate(dataset.subjects):
            print(f"subject {i+1:4d} of {len(dataset):4d}")
            affine = torch.tensor(nib.load(ds_dir / sub / img.filename).affine, dtype=torch.float)
            image, y_pred, adjusted_affine, _ = eval_step(None, dataset[i], affine)

            this_out = out_dir / dataset.name / sub

            match args.format:
                case "torch":
                    # surfaces are in the voxel space of the cropped image
                    write_surface_torch(
                        y_pred["surface"],
                        adjusted_affine[0],# @ torch.linalg.inv(affine),
                        this_out,
                        resolution=dataset.target_surface_resolution,
                    )
                case "freesurfer":
                    vol_info["volume"] = list(image.shape[-3:])
                    write_surface_freesurfer(y_pred["surface"], faces, this_out, adjusted_affine[0].numpy(), vol_info)

def write_surface_freesurfer(surfaces, faces, out_dir, affine, volume_info):
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    for hemi, s in surfaces.items():
        for surf, ss in s.items():
            # v = ss.vertices
            v = ss
            assert len(v) == 1
            v = v[0]
            v = v.cpu().numpy() @ affine[:3,:3].T + affine[:3,3]
            # faces = ss.faces.cpu().numpy()
            nib.freesurfer.write_geometry(out_dir / f"{hemi}.{surf}.pred", v, faces, volume_info=volume_info)


def write_surface_torch(
    surfaces: dict, affine: torch.Tensor, out_dir: Path, prefix=None, resolution=6, label="prediction", ext="pt"
):
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    resolution = str(resolution)
    for hemi, s in surfaces.items():
        for surf, ss in s.items():
            assert len(ss) == 1
            merge = [hemi, surf, resolution, label, ext]
            name = ".".join(merge if prefix is None else [prefix] + merge)
            v = ss[0].cpu()
            v = v @ affine[:3,:3].T + affine[:3,3]
            torch.save(v, out_dir / name)


def parse_args(argv):
    description = "Main interface to evaluating a BrainNet model."
    parser = argparse.ArgumentParser(
        prog="BrainNetEvaluator",
        description=description,
    )
    parser.add_argument(
        "config", help="Configuration file defining the parameters for training. This is used for setting up the model"
    )
    parser.add_argument(
        "checkpoint",
        default=None,
        type=int,
        help="Evaluate the model at checkpoint.",
    )
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--format", choices=["torch", "freesurfer"], default="freesurfer", help="Format in which to save predictions.")
    parser.add_argument("--subset", default="validation", type=str,
        help="Subset of data to evaluate on (e.g., train, validation, test)."
    )
    parser.add_argument("--datasets", default=None, nargs="+",
        help="Subset of data to evaluate on (e.g., train, validation, test)."
    )

    return parser.parse_args(argv[1:])

if __name__ == "__main__":
    args = parse_args(sys.argv)
    predict(args)
