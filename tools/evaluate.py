import argparse
import sys

from brainnet.evaluation import evaluate


def parse_args(argv):
    description = "Evaluate one or more checkpoints of a model."
    parser = argparse.ArgumentParser(prog="Evaluate a model", description=description)
    parser.add_argument("model", help="The model to evaluate.")
    parser.add_argument(
        "specs", help="Configuration file defining the parameters for training."
    )
    parser.add_argument(
        "subset",
        type=str,
        help="Subset of data to evaluate on (e.g., train, validation, test, exclude).",
    )
    parser.add_argument(
        "checkpoints",
        default=None,
        nargs="+",
        type=int,
        help="Evaluate the model at checkpoint.",
    )
    # parser.add_argument(
    #     "--datasets",
    #     default=None,
    #     nargs="+",
    #     help="Subset of data to evaluate on (e.g., ABIDE, HCP, ...).",
    # )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    evaluate(args.model, args.specs, args.subset, args.checkpoints)
