import argparse
import sys

from brainnet.evaluation import predict


from brainnet.command import handle_checkpoint_type


def parse_args(argv):
    description = "Main interface to evaluating a BrainNet model."
    parser = argparse.ArgumentParser(
        prog="BrainNetEvaluator",
        description=description,
    )
    help_model = "The model to predict."
    help_specs = "Configuration file defining the parameters for training."
    help_checkpoint = "Predict using this checkpoint."
    help_subset = "Subset of data to evaluate on (e.g., train, validation, test)."

    parser.add_argument("model", help=help_model)
    parser.add_argument("specs", help=help_specs)
    parser.add_argument("subset", type=str, help=help_subset)
    parser.add_argument("checkpoint", type=handle_checkpoint_type, help=help_checkpoint)
    parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        nargs="+",
        help="Subset of data to evaluate on (e.g., ABIDE, HCP, ...).",
    )
    parser.add_argument("--csv", default=None, type=str, help="")

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    predict(
        args.model, args.specs, args.subset, args.checkpoint, args.datasets, args.csv
    )
