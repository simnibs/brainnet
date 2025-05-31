import argparse
from pathlib import Path
import sys

import torch

from brainnet import resources_dir
from brainnet.config.topofit.train_parameters import TrainParameters


def dump_model(args):
    contrast = args.contrast
    resolution = args.resolution
    checkpoint = args.checkpoint
    out_dir = Path(args.out)

    print(f"Saving model configuration to {out_dir.resolve()}")

    p = TrainParameters(contrast=contrast, resolution=resolution)
    p.dump_prediction_config(out_dir)

    filename = p.results.checkpoint_dir / f"state_checkpoint_{checkpoint:05d}.pt"
    ckpt = torch.load(filename)["model"]
    torch.save(ckpt, out_dir / "t1w_1mm_state.pt")


def parse_args(argv):
    parser = argparse.ArgumentParser("Save TopoFit checkpoint for prediction.")
    parser.add_argument("contrast")
    parser.add_argument("resolution")
    parser.add_argument("checkpoint", type=int)
    parser.add_argument(
        "--out", "-o", default=str(resources_dir / "models" / "topofit")
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    dump_model(args)
