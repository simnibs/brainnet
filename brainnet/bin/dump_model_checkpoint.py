import argparse
import importlib
from pathlib import Path
import sys

import torch

from brainnet import RESOURCES_DIR

from brainnet.command import handle_checkpoint_type


def dump_model(args):
    if args.out is None:
        out_dir = RESOURCES_DIR / "models" / args.model
    else:
        out_dir = Path(args.out)

    print(f"Model : {args.model}")
    print(f"Saving configuration to {out_dir.resolve()}")

    train_parameters = importlib.import_module(
        f"brainnet.config.{args.model}.train_parameters"
    )

    p = train_parameters.TrainParameters(
        contrast=args.contrast, resolution=args.resolution, run_suffix=args.suffix
    )
    p.dump_prediction_config(out_dir)

    if args.checkpoint == "best":
        filename = p.results.checkpoint_dir / "state_checkpoint_best.pt"
        ckpt = torch.load(filename)
    else:
        filename = (
            p.results.checkpoint_dir / f"state_checkpoint_{args.checkpoint:05d}.pt"
        )
        ckpt = torch.load(filename)["model"]
    torch.save(ckpt, out_dir / f"{args.contrast}_{args.resolution}_state.pt")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Model state saver",
        description="Extract and save a model checkpoint and state which can be used for prediction.",
    )
    parser.add_argument("model")
    parser.add_argument("contrast")
    parser.add_argument("resolution")
    parser.add_argument("checkpoint", type=handle_checkpoint_type)
    parser.add_argument("--suffix", "-s", default="")
    parser.add_argument(
        "--out", "-o", help="Output directory in which to save the state."
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    dump_model(args)
