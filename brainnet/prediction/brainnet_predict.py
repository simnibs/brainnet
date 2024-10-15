import argparse
import importlib
from pathlib import Path
import sys

import torch
from ignite.engine import Engine

import brainnet.initializers
import brainnet.train.utilities
from brainnet.train.brainnet_train import EvaluationStep

def write_surface(
    surfaces: dict, out_dir: Path, prefix=None, resolution=5, label="pred", ext="pt"
):
    resolution = str(resolution)
    for hemi, s in surfaces.items():
        for surf, ss in s.items():
            assert len(v := ss.vertices) == 1
            merge = [hemi, surf, resolution, label, ext]
            name = ".".join(merge if prefix is None else [prefix] + merge)
            torch.save(v, out_dir / name)

def predict(args):

    """

    train_setup_file = "brainnet.config.cortex_mri.main"
    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")

    args = parse_args(["blabla", "brainnet.config.cortex.synth.main", "1200", "/mnt/scratch/personal/jesperdn/results/prediction"])

    """

    train_setup_file = args.config  # "brainnet.config.cortex.main"

    print("Setting up prediction...")

    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")
    train_setup.train_params.load_checkpoint = args.checkpoint

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    criterion = brainnet.initializers.init_criterion(train_setup.criterion)[args.subset]
    dataloader = brainnet.initializers.init_dataloader(
        train_setup.dataset, train_setup.dataloader
    )[args.subset]
    model = brainnet.initializers.init_model(train_setup.model)
    synth = brainnet.initializers.init_synthesizer(train_setup.synthesizer)[args.subset]

    eval_step = EvaluationStep(
        synth,
        model,
        criterion,
        enable_amp=train_setup.train_params.enable_amp,
    )
    evaluator = Engine(eval_step)

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


    dataset_idx = []
    subject_idx = []
    losses = []

    for dataset in dataloader.dataset.datasets:
        for i in range(len(dataset)):
            print(f"{dataset.name:20s} - {dataset.subject[i]:10s} : {loss['white']['chamfer']:5.2.f} {loss['pial']['chamfer']:5.2.f}")

            loss, image, y_pred, y_true = eval_step(evaluator, dataset[i])

            dataset_idx.append(dataset.name)
            subject_idx.append(dataset.subjects[i])
            losses.append(loss)

            this_out = out_dir / dataset.name / dataset.subjects[i]
            if not this_out.exists():
                this_out.mkdir(parents=True)

            match args.format:
                case "torch":
                    write_surface(
                        y_pred["surface"],
                        this_out,
                        # prefix=f"{dataset.name}.{dataset.subjects[i]}",
                        resolution=dataset.target_surface_resolution,
                        label="pred",
                        ext="pt"
                    )
                case "freesurfer":
                    raise NotImplementedError


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
    parser.add_argument("--format", choices=["torch", "freesurfer"], default="torch", help="Format in which to save predictions.")
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
