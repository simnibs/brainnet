import argparse
import sys

import brainnet.helpers
from brainnet.train import train


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="Train a model",
        description="Main interface to training a models.",
    )
    parser.add_argument(
        "model", help="Model to train.", choices=brainnet.helpers.__all__
    )

    parser.add_argument(
        "specs",
        help="Configuration file defining the parameters for training.",
    )
    # parser.add_argument(
    #     "--load-checkpoint",
    #     default=None,
    #     type=int,
    #     help="Resume training from this checkpoint.",
    # )
    # parser.add_argument(
    #     "--max-epochs",
    #     default=None,
    #     type=int,
    #     help="Terminate training when this number of epochs is reached.",
    # )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Disable logging with wandb.",
    )
    return parser.parse_args(argv[1:])


"""Example

python tools/train.py topofit synth_1mm --no-wandb

args = parse_args("brainnet/train/topofit.py t1w_1mm --no-wandb".split())

"""

if __name__ == "__main__":
    args = parse_args(sys.argv)
    # model_helpers = importlib.import_module(f"brainnet.helpers.{args.model}")
    model_helpers = getattr(brainnet.helpers, args.model)
    train(args.model, args.specs, model_helpers.create_trainer, args.no_wandb)
