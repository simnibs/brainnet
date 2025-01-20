import argparse
from pathlib import Path
from typing import Any, Callable
import torch

from ignite.engine import Engine, Events
from ignite.engine.events import CallableEventWithFilter
from ignite.handlers import ModelCheckpoint
from ignite.metrics.metric import Metric

import brainnet.config
from brainnet.config.base import EventAction
from brainnet import event_handlers
import brainnet.initializers
from brainnet.modules.criterion import CriterionAggregator


def print_memory_usage(device):
    # https://medium.com/deep-learning-for-protein-design/a-comprehensive-guide-to-memory-usage-in-pytorch-b9b7c78031d3
    total_memory = torch.cuda.get_device_properties(device).total_memory * 1e-9
    alloc = torch.cuda.memory_allocated(device) * 1e-9
    alloc_max = torch.cuda.max_memory_allocated(device) * 1e-9
    res = torch.cuda.memory_reserved(device) * 1e-9
    res_max = torch.cuda.max_memory_reserved(device) * 1e-9

    print("Memory [GB]        Current           Max")
    print("----------------------------------------")
    print(f"Total                            {total_memory:7.3f}")
    print(f"Allocated          {alloc:7.3f}       {alloc_max:7.3f}")
    print(f"Reserved           {res:7.3f}       {res_max:7.3f}")
    print("----------------------------------------")


def add_custom_events(engine, events: list[EventAction], other: Engine | None = None):
    """Set all custom events."""

    for e in events:
        engine.add_event_handler(e.event, e.handler, **e.kwargs)


def add_model_checkpoint(
    engine, to_save: dict[str, Any], config: brainnet.config.ResultsParameters
):
    """

    Parameters
    ----------
    engine
    to_save : dict
        Dictionary containing the items to save, e.g., model, optimizer,
        engine.
    config : brainnet.config.ResultsParameters
        Results configuration.
    """
    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        config.checkpoint_dir,
        config.checkpoint_prefix,
        n_saved=None,  # keep all
        require_empty=config.require_empty,
        filename_pattern=config.checkpoint_filename_pattern,
        # score_function=score_function,
        # score_name="accuracy",
        global_step_transform=lambda e, _: e.state.epoch,  # use epoch instead of iteration
    )

    engine.add_event_handler(config.save_checkpoint_on, model_checkpoint, to_save)


def load_checkpoint(to_load, train_setup):
    if train_setup.train_params.load_checkpoint != 0:
        ckpt_name = train_setup.results.checkpoint_filename_pattern.format(
            filename_prefix=train_setup.results.checkpoint_prefix,
            name="checkpoint",
            global_step=train_setup.train_params.load_checkpoint,
        )
        print(f"Loading checkpoint {ckpt_name}")
        ckpt = train_setup.results._from_checkpoint_dir / ckpt_name
        ckpt = torch.load(ckpt, map_location=train_setup.device)
        ModelCheckpoint.load_objects(to_load, ckpt)


def add_wandb_logger(engine, evaluators, config: brainnet.config.WandbParameters):
    """Logging with Wandb"""
    if not config.enable:
        return

    # wandb_dir = Path(config.wandb_dir)
    # if not wandb_dir.exists():
    #     wandb_dir.mkdir(parents=True)

    # like wandb.init()
    # logger = wandb_logger.WandBLogger(
    #     project=config.project,
    #     name=config.name,
    #     dir=wandb_dir,
    #     resume=config.resume,
    #     # **config.kwargs,
    #     # log the configuration of the run
    #     # config=recursive_namespace_to_dict(config),
    # )

    # # Log optimizer parameters
    # logger.attach_opt_params_handler(
    #     engine,
    #     event_name=config.log_on,
    #     optimizer=optimizer,
    #     param_name='lr'  # optional
    # )

    # # Log each evaluator
    # for k,v in evaluators.items():
    #     logger.attach_output_handler(
    #         v,
    #         event_name=config.log_on,
    #         tag=k,
    #         metric_names=["loss"],
    #         global_step_transform=global_step_from_engine(engine),
    #     )

    # engine.add_event_handler(Events.COMPLETED, logger.close)

    import wandb

    wandb_dir = Path(config.wandb_dir)
    if not wandb_dir.exists():
        wandb_dir.mkdir(parents=True)

    logger = wandb.init(
        project=config.project,
        name=config.name,
        dir=wandb_dir,
        resume=config.resume,
        id=config.run_id,
        tags=config.tags,
        # fork_from=config.fork_from,
        # log the configuration of the run
        # config=recursive_namespace_to_dict(config),
    )

    # Log the loss accumulated during training
    engine.add_event_handler(
        Events.EPOCH_COMPLETED, event_handlers.wandb_log_engine, logger, "trainer"
    )

    # Log the loss accumuated during evaluation
    for k, v in evaluators.items():
        engine.add_event_handler(
            config.log_on, event_handlers.wandb_log_evaluator, logger, k, v
        )

    # Add an event that closes the logger on completion
    engine.add_event_handler(Events.COMPLETED, event_handlers.wandb_finish, logger)


def add_terminal_logger(engine):
    """Logging to terminal."""
    engine.add_event_handler(Events.EPOCH_COMPLETED, event_handlers.log_epoch)


# def add_state_entries(engine, config):
#     """Set additional state variables besides the default ones.

#     References
#     ----------
#     https://pytorch.org/ignite/_modules/ignite/engine/events.html#State
#     """
#     setattr(engine.state, "evaluation", {})
#     setattr(engine.state, "config", config)
#     setattr(
#         engine.state,
#         "head_runtime_kwargs",
#         {
#             k: recursive_namespace_to_dict(v.runtime_kwargs)
#             for k, v in vars(config.model.heads).items()
#             if hasattr(v, "runtime_kwargs")
#         },
#     )

#     engine.state._update_attrs()


def write_example_to_disk(
    engine: Engine,
    evaluators: dict[str, Engine],
    config: brainnet.config.ResultsParameters,
    writer=event_handlers.write_example,
):
    engine.add_event_handler(
        config.save_example_on,
        writer,
        evaluators=evaluators,
        config=config,
    )

def add_metric_to_engine(
        engine,
        metric: Metric = CriterionAggregator(),
        name: str = "loss"
    ):
    metric.attach(engine, name)


def add_evaluation_event(
    eval_step,
    engine: Engine,
    dataloader,
    evaluate_on: CallableEventWithFilter,
    logger: Callable,
    epoch_length: int | None,
) -> Engine:

    metric = CriterionAggregator()

    evaluator = Engine(eval_step)

    # Add an event that synchronizes iteration and epoch count from the trainer
    # evaluator.add_event_handler(
    #     Events.STARTED,
    #     event_handlers.synchronize_state,
    #     other=engine,
    #     attrs=["iteration", "epoch"],
    # )

    # attach metric to the evaluator
    metric.attach(evaluator, "loss")

    # add an event to TRAINER that performs the evaluation
    engine.add_event_handler(
        evaluate_on,
        event_handlers.evaluate_model,
        evaluator=evaluator,
        dataloader=dataloader,
        epoch_length=epoch_length or len(iter(dataloader)),
        logger=logger,
    )

    return evaluator


def parse_args(argv):
    description = "Main interface to training a BrainNet model. For convenience, a few parameters are exposed on the command line. Values provided here will overwrite those set in the configuration file."
    parser = argparse.ArgumentParser(
        prog="BrainNetTrainer",
        description=description,
    )
    parser.add_argument(
        "config", help="Configuration file defining the parameters for training."
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        type=int,
        help="Resume training from this checkpoint.",
    )
    parser.add_argument(
        "--max-epochs",
        default=None,
        type=int,
        help="Terminate training when this number of epochs is reached.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Disable logging with wandb.",
    )

    return parser.parse_args(argv[1:])
