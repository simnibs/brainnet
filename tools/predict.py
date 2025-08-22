import argparse
from pathlib import Path
import sys

import brainnet.helpers


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="brainnet",
        description="Main interface to prediction using the BrainNet models.",
    )

    # The programs
    networks = parser.add_subparsers(help="Available networks.")
    topofit = networks.add_parser("topofit", help="Fit cortical surfaces to MRI scans.")
    trega = networks.add_parser(
        "trega",
        help="Template REGistration (Affine). Affine transformation between subject and MNI space.",
    )
    # tregn = parser.add_subparser(
    #     "tregn",
    #     "Template REGistration (Nonlinear). Nonlinear transformation between subject and MNI space.",
    # )

    # Arguments common to all programs
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a single image or a text file containing a list of filenames of images.",
    )
    parser.add_argument(
        "out",
        type=Path,
        help=(
            "Path to a directory or a text file containing a list of "
            "directories in which to store the surface predictions."
        ),
    )

    parser.add_argument(
        "--contrast", "-c", choices=["t1w", "synth"], default="t1w", help=""
    )
    parser.add_argument(
        "--resolution", "-r", choices=["1mm", "random"], default="1mm", help=""
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

    # Arguments specific to subprograms

    # TOPOFIT
    topofit.add_argument(
        "-t",
        "--transform",
        type=Path,
        help=(
            "Path to a text file containing a single MNI transformation or a "
            "text file containing a list of filenames of MNI transformations."
        ),
    )
    topofit.add_argument(
        "--mni-direction",
        choices=["mni2sub", "sub2mni"],
        default="mni2sub",
        help="Direction of MNI transformation.",
    )
    topofit.add_argument(
        "--mni-space",
        choices=["mni152", "mni305"],
        default="mni152",
        help="MNI space to which the transform relates.",
    )
    topofit.add_argument(
        "--hemi",
        choices=["lh", "rh"],
        default="both",
        help="Hemisphere to predict. Default is both.",
    )
    topofit.set_defaults(func=brainnet.helpers.topofit.predict)

    # NORM AFFINE
    trega.add_argument(
        "--mni-space",
        choices=["mni152", "mni305"],
        default="mni152",
        help="MNI space to which the transform relates.",
    )
    trega.set_defaults(func=brainnet.helpers.trega.predict)

    # NORM NONLIN
    # tregn.add_argument(
    #     "--mni-space",
    #     choices=["mni152", "mni305"],
    #     default="mni152",
    #     help="MNI space to which the transform relates.",
    # )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    args.func(args)
