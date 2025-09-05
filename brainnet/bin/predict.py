import argparse
from pathlib import Path
import sys

import brainnet.helpers


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="brainnet",
        description="Main interface to prediction using the BrainNet models.",
    )
    networks = parser.add_subparsers(help="Available networks.")

    # Arguments common to all programs
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a single image or a text file containing a list of filenames of images.",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help=(
            "Path to a directory or a text file containing a list of "
            "directories in which to store the surface predictions."
        ),
    )

    parser.add_argument(
        "--conform",
        action="store_true",
        help=(
            "Whether or not to conform (resample to 1 mm resolution and align "
            "with identity affine [RAS]) the image before prediction."
        ),
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        help="The device on which to run the predictions.",
    )

    # The programs
    topofit = networks.add_parser(
        "topofit", help="Estimate cortical surfaces on an MRI scan using TopoFit."
    )
    trega = networks.add_parser(
        "trega",
        description=brainnet.helpers.trega.DESCRIPTION,
        help="Estimate an affine transformation between subject and MNI space.",
    )
    # tregn = parser.add_subparser(
    #     "tregn",
    #     "Template REGistration (Nonlinear). Nonlinear transformation between subject and MNI space.",
    # )

    # Arguments specific to subprograms

    # TOPOFIT
    topofit.add_argument(
        "--contrast",
        "-c",
        choices=["t1w", "synth"],
        default="t1w",
        help="Image modality that the model was trained on (default = t1w)",
    )
    topofit.add_argument(
        "--resolution",
        "-r",
        choices=["1mm", "random"],
        default="1mm",
        help="Image resolution that the model was trained on (default = 1mm)",
    )
    topofit.add_argument(
        "-t",
        "--transform",
        type=Path,
        help=(
            "Path to a text file containing a single MNI transformation or a "
            "text file containing a list of filenames pointing to such text "
            "files."
        ),
    )
    topofit.add_argument(
        "--mni-direction",
        choices=["mni2sub", "sub2mni"],
        default="",
        help="Direction of the supplied MNI transformation (default = mni2sub). Only used if -t/--transforms is specified.",
    )
    topofit.add_argument(
        "--mni-space",
        choices=["mni152", "mni305"],
        default="mni152",
        help="MNI space to which the transform relates (default = mni152). Only used if -t/--transforms is specified.",
    )
    topofit.add_argument(
        "--hemi",
        choices=["lh", "rh"],
        default="both",
        help="Hemisphere to predict (default = both).",
    )
    topofit.set_defaults(func=brainnet.helpers.topofit.predict)

    # AFFINE NORMALIZATION
    trega.add_argument(
        "--mni-space",
        choices=["mni152", "mni305"],
        default="mni152",
        help="Return a transformation to this MNI space (default = mni152).",
    )
    trega.add_argument(
        "--all",
        action="store_true",
        help=(
            "The network estimates three transformations: one for left and "
            "right hemisphere and one for the entire brain. By default, only "
            "the transformation estimated for the whole brain is written to "
            "disk. Setting this flag will write all transformations instead."
        ),
    )
    trega.set_defaults(func=brainnet.helpers.trega.predict)

    # NONLINEAR NORMALIZATION
    # tregn.add_argument(
    #     "--mni-space", **MNI_SPACE)

    return parser.parse_args(argv[1:])


def main():
    args = parse_args(sys.argv)
    args.func(args)


if __name__ == "__main__":
    main()
