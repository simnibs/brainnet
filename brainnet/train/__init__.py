import importlib
from typing import Callable

__all__ = ["import_train_parameters", "train"]


def import_train_parameters(model):
    return importlib.import_module(".train_parameters", f"brainnet.config.{model}")


def train(
    model: str, specs_name: str, create_trainer: Callable, no_wandb: bool = False
):
    """

    python brainnet/train/topofit.py synth_1mm --no-wandb

    args = brainnet.train.utilities.argparser_topofit(
        "brainnet/train/topofit.py t1w_1mm --no-wandb".split()
    )

    """

    print(f"Using training specs: {specs_name}", end="\n\n")

    train_parameters = import_train_parameters(model)
    specs = importlib.import_module(f"brainnet.config.{model}.{specs_name}")

    try:
        PHASES = dict(OVERRIDE=getattr(specs, "OVERRIDE"))
        PHASES = specs.PHASES if PHASES is None else PHASES
    except AttributeError:
        PHASES = specs.PHASES

    for name, phase in PHASES.items():
        print(f"STARTING TRAINING PHASE: {name}")
        print(79 * "=")
        print(f"Specification\n    {phase}")
        print(f"Defaults\n    {specs.DEFAULTS}")

        setup = train_parameters.TrainParameters(**(specs.DEFAULTS | phase))
        trainer, dataloader = create_trainer(setup, no_wandb)
        trainer.run(
            dataloader,
            epoch_length=setup.trainer_epoch_length or len(iter(dataloader)),
            max_epochs=setup.max_epochs,
        )
        print(f"TRAINING PHASE DONE: {name}", end="\n\n")
        print(79 * "=")
