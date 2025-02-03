import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch

from brainsynth.dataset import PredictionDataset
from brainnet.prediction import PretrainedModels
from brainnet.prediction.brainnet_predict import PredictionStep

vol_info = dict(
    head=[2, 0, 20],
    valid="1  # volume info valid",
    filename="vol.nii",
    volume=[256, 256, 256],
    voxelsize=[1, 1, 1],
    xras=[-1, 0, 0],
    yras=[0, 0, -1],
    zras=[0, 1, 0],
    cras=[0, 0, 0],
)


def predict(args):
    device = torch.device(args.device)

    try:
        # single image filename
        _ = nib.load(args.image)
        images = [args.image]
    except nib.filebasedimages.ImageFileError:
        # list of image filenames
        images = np.loadtxt(args.image, dtype=str)

    try:
        # a single transform
        t = np.loadtxt(args.transform)
        assert t.shape == (4, 4)
        transforms = [args.transform]
    except ValueError:
        # list of transform filenames
        transforms = np.loadtxt(args.transform, dtype=str)

    try:
        out_dir = np.loadtxt(args.out, dtype=str)
    except (FileNotFoundError, IsADirectoryError):
        out_dir = [args.out]

    dataset = PredictionDataset(
        images=images,
        mni_transforms=transforms,
        mni_direction=args.mni_dir,
        mni_space=args.mni_space,
        conform=args.conform,
    )

    print("|========================================|")
    print("| Topofit                                |")
    print("|                                        |")
    print("| Specifications                         |")
    print(f"|   Contrast     : {args.contrast:>10s}            |")
    print(f"|   Resolution   : {args.resolution:>10s}            |")
    print("|========================================|")
    print()

    specs = (args.contrast, args.resolution)
    pretrained_models = PretrainedModels()
    model = pretrained_models.load_model("topofit", specs, device)
    preprocessor = pretrained_models.load_preprocessor("topofit", specs, device)
    predict_step = PredictionStep(preprocessor, model, enable_amp=True)
    faces = {h: t.faces.cpu().numpy() for h, t in predict_step.topology.items()}

    print("Processing subjects")
    n = len(dataset)
    nchar = len(str(n))
    for i, batch in enumerate(dataset):
        print(f"{i + 1:{nchar}d} of {n:d}")

        y_pred, _ = predict_step(None, batch)
        y_pred = y_pred["surface"]
        out = Path(out_dir[i])
        if not out.exists():
            out.mkdir(parents=True)

        for h, surfaces in y_pred.items():
            for s, vertices in surfaces.items():
                nib.freesurfer.write_geometry(
                    out / f"{h}.{s}",
                    vertices.cpu().numpy(),
                    faces[h],
                    volume_info=vol_info,
                )


def parse_args(argv):
    description = "Main interface to prediction using a BrainNet model."
    parser = argparse.ArgumentParser(
        prog="BrainNetPrediction",
        description=description,
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to a single image or a text file containing a list of filenames of images.",
    )
    parser.add_argument(
        "transform",
        type=str,
        help="Path to a text file containing a single MNI transformation or a text file containing a list of filenames of MNI transformations",
    )
    parser.add_argument(
        "out",
        type=str,
        help="Path to a directory or a text file containing a list of directories in which to store the surface predictions.",
    )
    parser.add_argument(
        "--mni-dir",
        choices=["mni2sub", "sub2mni"],
        default="mni2sub",
        help="Direction of MNI transformation.",
    )
    parser.add_argument(
        "--mni-space",
        choices=["mni152", "mni305"],
        default="mni152",
        help="MNI space to which the transform relates.",
    )
    parser.add_argument(
        "--conform",
        action="store_true",
        help="Whether or not to conform (resample to 1 mm resolution, align with identity affine) the image before prediction.",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        help="The device on which to run the predictions.",
    )
    parser.add_argument(
        "--contrast", "-c", choices=["t1w", "synth"], default="t1w", help=""
    )
    parser.add_argument(
        "--resolution", "-r", choices=["1mm", "random"], default="1mm", help=""
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    predict(args)
