import argparse
import importlib
import sys

from brainnet.train import train


# def parse_args(argv):
#     description = "Main interface to training a BrainNet model. For convenience, a few parameters are exposed on the command line. Values provided here will overwrite those set in the configuration file."
#     parser = argparse.ArgumentParser(
#         prog="BrainNetTrainer",
#         description=description,
#     )
#     parser.add_argument(
#         "config", help="Configuration file defining the parameters for training."
#     )
#     parser.add_argument(
#         "--load-checkpoint",
#         default=None,
#         type=int,
#         help="Resume training from this checkpoint.",
#     )
#     parser.add_argument(
#         "--max-epochs",
#         default=None,
#         type=int,
#         help="Terminate training when this number of epochs is reached.",
#     )
#     parser.add_argument(
#         "--no-wandb",
#         action="store_true",
#         default=False,
#         help="Disable logging with wandb.",
#     )
#     # parser.add_argument(
#     #     "--resume",
#     #     type=str,
#     #     default=None,
#     #     help="Resume from run.",
#     # )

#     return parser.parse_args(argv[1:])


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="Train a model",
        description="Main interface to training a models.",
    )
    parser.add_argument("model", help="Model to train.")

    parser.add_argument(
        "specs", help="Configuration file defining the parameters for training."
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

python brainnet/train/topofit.py synth_1mm --no-wandb

args = parse_args("brainnet/train/topofit.py t1w_1mm --no-wandb".split())

"""

if __name__ == "__main__":
    args = parse_args(sys.argv)
    model_helpers = importlib.import_module(f"brainnet.helpers.{args.model}")
    train(args.model, args.specs, model_helpers.create_trainer, args.no_wandb)
