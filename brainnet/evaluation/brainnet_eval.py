import argparse
import importlib
import sys
import time

import pandas as pd

import brainsynth.dataset

import brainnet.initializers
import brainnet.train.utilities
from brainnet.evaluation.utilities import MetricAggregator


def create_evaluator(model, setup, subset: str = "validation"):
    train = importlib.import_module(f".{model}", "brainnet.train")

    criterion = brainnet.initializers.init_criterion(setup.criterion)[subset]
    model = train.setup_model(setup)
    synth = brainnet.initializers.init_synthesizer(setup.synthesizer)[subset]

    dataloaders = brainsynth.dataset.setup_dataloader(
        setup.dataset[args.subset],
        separate_datasets=True,
        **setup.dataloader,
    )

    eval_step = train.EvaluationStep(
        synth,
        model,
        criterion,
        enable_amp=setup.enable_amp,
    )
    # evaluator = Engine(eval_step)

    to_load = dict(
        model=model,
        **{f"criterion[{subset}]": criterion},
    )
    brainnet.train.utilities.load_checkpoint_from_setup(to_load, setup)

    # Write the collected metrics to this directory
    out_dir = setup.results.evaluation_dir  # / args.subset

    print("Setup completed.")
    print(setup)

    print("Evaluation settings")
    print(f"  Output dir    {out_dir}")
    print(f"  Subset        {subset}", end="\n\n")

    return eval_step, dataloaders, out_dir


def get_setup_at_checkpoint(specs_name, model, checkpoint):
    train_parameters = importlib.import_module(
        ".train_parameters", f"brainnet.config.{model}"
    )
    specs = importlib.import_module(f".{specs_name}", f"brainnet.config.{model}")

    try:
        phase = getattr(specs, "OVERRIDE")
        name = "OVERRIDE"
    except AttributeError:
        PHASES = specs.PHASES
        for k, v in PHASES.items():
            start = v["load_checkpoint"] if "load_checkpoint" in v else 0
            if start < checkpoint <= v["max_epochs"]:
                name = k
                phase = v
                break
    phase["load_checkpoint"] = checkpoint

    print(79 * "=")
    print(f"Evaluating model : {model}")
    print(f"  Specs          : {specs_name}")
    print(f"  Phase          : {name}")
    print(f"Specification\n    {phase}")
    print(f"Defaults\n    {specs.DEFAULTS}")
    print(79 * "=", end="\n\n")

    return train_parameters.TrainParameters(**(specs.DEFAULTS | phase))


def evaluate_checkpoint(checkpoint, model, subset, specs_name):
    setup = get_setup_at_checkpoint(specs_name, model, checkpoint)

    eval_step, dataloaders, out_dir = create_evaluator(model, setup, subset)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    metric = MetricAggregator()

    # Save losses and corresponding subject IDs
    t_total = 0.0
    df_metric = []
    for k, v in dataloaders.items():
        t_start = time.perf_counter()

        dataset = v.dataset
        print(f"{k:<20s} [n = {len(v):4d}]", end="", flush=True)

        for batch in dataset:
            # loss, image[, vox2mri], y_pred, y_true
            out = eval_step(None, batch)
            loss = out[0]
            metric.update([loss])

        index = pd.MultiIndex.from_product([[k], dataset.subjects])
        df = metric.compute(index)
        df_metric.append(df)
        metric.reset()

        t_stop = time.perf_counter()
        t_elapse = t_stop - t_start
        t_total += t_elapse
        print(f"    ({t_elapse:7.2f} s)")

    df_metric = pd.concat(df_metric)
    df_metric.to_pickle(out_dir / f"{args.subset}-checkpoint-{checkpoint:05d}.pickle")

    print(f"Total time to evaluate {t_total:7.2f} s")


def evaluate(args):
    """

    python brainnet/evaluation/brainnet_eval.py brainnet.config.topofit.adapt.main_N10 1420 --subset validation --separate-evaluation

    args = parse_args("brainnet/evaluation/brainnet_eval.py topofit t1w_1mm validation 580 600".split())
    args = parse_args("brainnet/evaluation/brainnet_eval.py alignment t1w_1mm validation 600".split())

    """

    for checkpoint in args.checkpoints:
        print(f"Evaluating checkpoint {checkpoint:05d}")
        evaluate_checkpoint(checkpoint, args.model, args.subset, args.specs)


def parse_args(argv):
    description = "Main interface to evaluating a BrainNet model."
    parser = argparse.ArgumentParser(
        prog="BrainNetEvaluator",
        description=description,
    )
    parser.add_argument("model", help="The model to evaluate (e.g., topofit).")
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
    parser.add_argument(
        "--datasets",
        default=None,
        nargs="+",
        help="Subset of data to evaluate on (e.g., ABIDE, HCP, ...).",
    )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    evaluate(args)
