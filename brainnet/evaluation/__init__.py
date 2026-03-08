import csv
import importlib
import itertools
import time

import pandas as pd

import brainsynth
from brainsynth.config.synthesizer import PredictionConfig
import brainsynth.dataset

import brainnet.initializers
import brainnet.helpers.utils
from brainnet.evaluation.utilities import MetricAggregator


def restrict_datasets_(setup, subset, datasets=None):
    if datasets is not None:
        setup.dataset[subset].dataset_kwargs = {
            k: v
            for k, v in setup.dataset[subset].dataset_kwargs.items()
            if k in datasets
        }
    assert len(setup.dataset[subset].dataset_kwargs) > 0


def create_evaluator(
    model, setup, subset: str = "validation", datasets: list[str] | tuple | None = None
):
    model_helper = importlib.import_module(f"brainnet.helpers.{model}")

    criterion = brainnet.initializers.init_criterion(setup.criterion)[subset]
    model = model_helper.setup_model(setup)
    synth = brainnet.initializers.init_synthesizer(setup.synthesizer)[subset]

    restrict_datasets_(setup, subset, datasets)
    dataloaders = brainsynth.dataset.setup_dataloader(
        setup.dataset[subset], separate_datasets=True, **setup.dataloader
    )

    kwargs = dict(enable_amp=setup.enable_amp, device=setup.device)

    if hasattr(setup, "iterative_hemisphere_prediction"):
        kwargs["iterative_hemisphere_prediction"] = (
            setup.iterative_hemisphere_prediction
        )
    eval_step = model_helper.EvaluationStep(synth, model, criterion, **kwargs)
    # evaluator = Engine(eval_step)

    to_load = dict(model=model, **{f"criterion[{subset}]": criterion})
    brainnet.helpers.utils.load_checkpoint_from_setup(to_load, setup)

    # Do not calculate SIF
    eval_step.criterion.update_loss_weights(
        {("white", "sif"): 0.0, ("pial", "sif"): 0.0}
    )

    # Write the collected metrics to this directory
    out_dir = setup.results.evaluation_dir

    print("Setup completed.")
    print(setup)

    print()
    print("Evaluation settings")
    print(f"  Output dir        {out_dir}")
    print(f"  Subset            {subset}", end="\n\n")

    return eval_step, dataloaders, out_dir


def create_predictor(
    model, setup, subset: str = "validation", datasets: list[str] | tuple | None = None
):
    model_helper = importlib.import_module(f"brainnet.helpers.{model}")
    model = model_helper.setup_model(setup)

    _valid_pred_contrasts = {"t1w", "t2w", "flair", "ct"}

    # We need the affine as well
    restrict_datasets_(setup, subset, datasets)
    for v in setup.dataset[subset].dataset_kwargs.values():
        found = [i for i in _valid_pred_contrasts if i in v["images"]]
        n_found = len(found)
        msg = f"{n_found} contrasts specified in dataset configuration ({found}) which is ambiguous for prediction. Need only one."
        assert n_found == 1, msg
        v["images"] = found

    dataloaders = brainsynth.dataset.setup_dataloader(
        setup.dataset[subset],
        separate_datasets=True,
        **setup.dataloader,
        dataset_class=brainsynth.dataset.SynthesizedDataset,
    )
    if subset in ("test", "ood", "exclude"):
        subset = "validation"
        # setup.prediction_config["preprocessor"]
    in_res = setup.synthesizer[subset].in_res
    out_size = setup.synthesizer[subset].out_size
    out_center_str = setup.synthesizer[subset].out_center_str
    preprocessor = brainsynth.Synthesizer(
        PredictionConfig("PredictionBuilder", in_res, out_size, out_center_str)
    )

    pred_step = model_helper.PredictionStep(
        None,
        preprocessor,
        model,
        **setup.prediction_config["step"],
        enable_amp=setup.enable_amp,
        device=setup.device,
    )
    write_step = model_helper.write_example_prediction

    to_load = dict(model=model)
    brainnet.helpers.utils.load_checkpoint_from_setup(to_load, setup)

    print("Setup completed.")
    print(setup)

    return pred_step, write_step, dataloaders


def get_setup_at_checkpoint(specs_name, model, checkpoint):
    tp = importlib.import_module(f"brainnet.config.{model}.train_parameters")
    specs = importlib.import_module(f"brainnet.config.{model}.{specs_name}")

    try:
        phase = getattr(specs, "OVERRIDE")
        name = "OVERRIDE"
    except AttributeError:
        PHASES = specs.PHASES
        if checkpoint == "best":
            # Assume that best is always from the last training phase
            name = tuple(PHASES.keys())[-1]
            phase = tuple(PHASES.values())[-1]
        else:
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

    return tp.TrainParameters(**(specs.DEFAULTS | phase))


def evaluate_checkpoint(model, specs_name, subset, checkpoint, datasets=None):
    setup = get_setup_at_checkpoint(specs_name, model, checkpoint)

    eval_step, dataloaders, out_dir = create_evaluator(model, setup, subset, datasets)
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
    df_metric.to_pickle(out_dir / f"{subset}-checkpoint-{checkpoint:05d}.pickle")

    print(f"Total time to evaluate {t_total:7.2f} s")


def evaluate(
    model, specs, subset, checkpoints, datasets: list[str] | tuple | None = None
):
    """Evaluate several checkpoints.

    Parameters
    ----------
    model : _type_
        _description_
    specs : _type_
        _description_
    subset : _type_
        _description_
    checkpoints : _type_
        _description_
    datasets : list[str] | tuple | None, optional
        _description_, by default None
    """
    for checkpoint in checkpoints:
        print(f"Evaluating checkpoint {checkpoint:05d}")
        evaluate_checkpoint(model, specs, subset, checkpoint, datasets)


def predict(
    model,
    specs,
    subset,
    checkpoint,
    datasets: list[str] | tuple | None = None,
    csv_file: str | None = None,
):
    """_summary_

    Parameters
    ----------
    model : _type_
        _description_
    specs : _type_
        _description_
    subset : _type_
        _description_
    checkpoint : _type_
        _description_
    datasets : list[str] | tuple | None, optional
        _description_, by default None
    csv_file : str | None, optional
        _description_, by default None
    """
    if csv_file is not None:
        assert datasets is None, "Cannot specify `datasets` when using `csv_file`."

    setup = get_setup_at_checkpoint(specs, model, checkpoint)
    pred_step, write_step, dataloaders = create_predictor(
        model, setup, subset, datasets
    )

    if isinstance(checkpoint, int):
        ckpt_name = f"checkpoint-{checkpoint:05d}"
    else:
        ckpt_name = f"checkpoint-{checkpoint:s}"
    out_dir = setup.results.evaluation_dir / subset / ckpt_name
    print(f"Output dir      {out_dir}")

    if csv_file is None:
        content = list(
            itertools.chain(
                *[
                    itertools.product([k], map(str, v.dataset.subjects))
                    for k, v in dataloaders.items()
                ]
            )
        )
    else:
        with open(csv_file, "r") as f:
            csv_reader = csv.reader(f)
            content = [row for row in csv_reader]

    subid2index = {
        k: dict(zip(map(str, v.dataset.subjects), range(len(v.dataset))))
        for k, v in dataloaders.items()
    }
    for dataset, subject in content:
        print(f"{dataset:15s} {subject:10s}", end="")

        out_dir_sub = out_dir / dataset / subject
        if not out_dir_sub.exists():
            out_dir_sub.mkdir(parents=True)

        ds = dataloaders[dataset].dataset
        i = subid2index[dataset][subject]

        contrast = ds.images[0]  # from create_predictor, this has length 1

        # The PredictionBuilder expects an image (and affine) called `image`
        # (which is used for prediction)
        batch = ds[i]
        images = batch[0][contrast]
        vox2ras = batch[1][contrast]
        template = batch[2]["template"]
        batch = (images, vox2ras, template)

        t_start = time.perf_counter()
        y_pred = pred_step(None, batch)
        t_elapse = time.perf_counter() - t_start
        print(f"    ({t_elapse:7.2f} s)")

        write_step(y_pred, out_dir_sub)
