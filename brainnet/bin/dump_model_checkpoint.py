import argparse
import importlib
from pathlib import Path
import sys
import warnings

import torch

from brainnet.resources import RESOURCES_DIR

from brainnet.command import handle_checkpoint_type


def dump_model(args):
    if args.out is None:
        out_dir = RESOURCES_DIR / "models" / args.model
    else:
        out_dir = Path(args.out)

    print(f"Model : {args.model}")
    print(f"Specs : {args.specs}")

    print(f"Saving configuration to {out_dir.resolve()}")

    train_parameters = importlib.import_module(
        f"brainnet.config.{args.model}.train_parameters"
    )
    specs = importlib.import_module(f"brainnet.config.{args.model}.{args.specs}")

    if args.suffix is not None:
        if specs.DEFAULTS["run_suffix"] is not None:
            warnings.warn(
                f"Overwriting `run_suffix` ({specs.DEFAULTS['run_suffix']} -> {args.suffix})"
            )
        specs.DEFAULTS["run_suffix"] = args.suffix

    p = train_parameters.TrainParameters(**specs.DEFAULTS)
    filename_config = out_dir / f"config_{args.specs}.json"
    print(f"Saving {filename_config}")
    p.dump_prediction_config(filename_config)

    if args.checkpoint == "best":
        filename = p.results.checkpoint_dir / "state_checkpoint_best.pt"
        ckpt = torch.load(filename)
    else:
        filename = (
            p.results.checkpoint_dir / f"state_checkpoint_{args.checkpoint:05d}.pt"
        )
        ckpt = torch.load(filename)["model"]
    print(f"Loaded checkpoint: {filename}")

    filename_state = out_dir / f"state_{args.specs}.pt"
    print(f"Saving {filename_state}")
    torch.save(ckpt, filename_state)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Model state saver",
        description="Extract and save a model checkpoint and state which can be used for prediction.",
    )
    parser.add_argument("model")
    parser.add_argument("specs")
    parser.add_argument("checkpoint", type=handle_checkpoint_type)
    parser.add_argument("--suffix", "-s", default=None)
    parser.add_argument(
        "--out", "-o", help="Output directory in which to save the state."
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    dump_model(args)
