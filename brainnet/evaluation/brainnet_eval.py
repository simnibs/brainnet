import argparse
import importlib
import sys

from ignite.engine import Engine

import brainsynth.dataset

import brainnet.initializers
import brainnet.train.utilities
from brainnet.train.brainnet_train import EvaluationStep
from brainnet.evaluation.utilities import add_metric_writer, MetricAggregator


def evaluate(args):
    """

    python brainnet/evaluation/brainnet_eval.py brainnet.config.topofit.adapt.main_N10 1420 --subset validation --separate-evaluation

    train_setup_file = "brainnet.config.topofit.synth.main"
    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")

    """

    train_setup_file = args.config

    print("Setting up evaluation...")

    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")
    train_setup.train_params.load_checkpoint = args.checkpoint

    # Write the collected metrics to this directory
    out_dir = train_setup.results.evaluation_dir / args.subset

    criterion = brainnet.initializers.init_criterion(train_setup.criterion)[args.subset]

    model = brainnet.initializers.init_model(train_setup.model)
    synth = brainnet.initializers.init_synthesizer(train_setup.synthesizer)[args.subset]

    eval_step = EvaluationStep(
        synth,
        model,
        criterion,
        enable_amp=train_setup.train_params.enable_amp,
    )
    evaluator = Engine(eval_step)

    # The order in which the events are added to the engine is important!

    to_load = dict(
        model=model,
        **{f"criterion[{args.subset}]": criterion},
    )
    brainnet.train.utilities.load_checkpoint(to_load, train_setup)

    print(f"Setup completed. Evaluating at epoch {args.checkpoint}")

    sep_line = 79 * "="

    print(sep_line)
    print(f"Config file     {train_setup_file}")
    print(f"Project         {train_setup.project:30s}")
    print(f"Run             {train_setup.run:30s}")
    print("Evaluation settings")
    print(f"  Output dir    {out_dir}")
    print(f"  Subset        {args.subset}")
    print(sep_line)

    # Start the training
    metric = MetricAggregator()

    if args.separate_evaluation:
        dataloaders = brainsynth.dataset.setup_dataloader(
            getattr(train_setup.dataset, args.subset),
            vars(train_setup.dataloader),
            separate_datasets=True,
        )
        for k, v in dataloaders.items():
            print(f"Evaluating on {k:s}")
            metric_name = f"loss-{k}"
            metric.attach(evaluator, metric_name)
            add_metric_writer(evaluator, out_dir, metric_name)
            evaluator.run(v, max_epochs=1)
            # detach metric and add again in next iteration with different name
            metric.detach(evaluator)
    else:
        dataloader = brainnet.initializers.init_dataloader(
            train_setup.dataset, train_setup.dataloader
        )[args.subset]

        # Aggregate losses and write at the end of epoch
        metric_name = "loss"
        metric.attach(evaluator, metric_name)
        add_metric_writer(evaluator, out_dir, metric_name)

        evaluator.run(dataloader, max_epochs=1)


def parse_args(argv):
    description = "Main interface to evaluating a BrainNet model."
    parser = argparse.ArgumentParser(
        prog="BrainNetEvaluator",
        description=description,
    )
    parser.add_argument(
        "config",
        help="Configuration file defining the parameters for training. This is used for setting up the model",
    )
    parser.add_argument(
        "checkpoint",
        default=None,
        type=int,
        help="Evaluate the model at checkpoint.",
    )
    parser.add_argument(
        "--subset",
        default="validation",
        type=str,
        help="Subset of data to evaluate on (e.g., train, validation, test).",
    )
    parser.add_argument(
        "--separate-evaluation",
        default=False,
        action="store_true",
        help="Evaluate on each dataset separately rather than across all selected datasets.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        nargs="+",
        help="Subset of data to evaluate on (e.g., train, validation, test).",
    )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    evaluate(args)
