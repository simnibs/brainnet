import argparse
import copy
import importlib
from pathlib import Path
import sys
from typing import Any, Callable
import torch

from ignite.engine import Engine, Events
from ignite.engine.events import CallableEventWithFilter
from ignite.handlers import ModelCheckpoint

import brainsynth

import brainnet.config
from brainnet.config.base import EventAction
from brainnet import event_handlers
import brainnet.initializers
from brainnet.mesh.surface import TemplateSurfaces
from brainnet.modules.brainnet import BrainNet
from brainnet.modules.head import SurfaceModule
from brainnet.modules.criterion import Criterion, CriterionAggregator
from brainnet.utilities import recursive_dict_sum


# points = np.concatenate((w.vertices[0].cpu().numpy(), p.vertices[0].cpu().numpy()))
# cells = np.concatenate(
#     (
#         np.full((w.topology.n_vertices, 1), 2),
#         np.ascontiguousarray(np.arange(len(points)).reshape(2, -1).T),
#     ),
#     axis=1,
# )
# lines = pv.PolyData(points, cells)

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


def recursive_itemize(d, out=None):
    """Recursively call .item() on values."""
    if out is None:
        out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {}
            recursive_itemize(v, out[k])
        else:
            out[k] = v.item()
    return out


def add_custom_events(engine, events: list[EventAction], other: Engine | None = None):
    """Set all custom events."""

    for e in events:
        engine.add_event_handler(e.event, e.handler, **e.kwargs)


def add_model_checkpoint(engine, to_save: dict[str, Any], config: brainnet.config.ResultsParameters):
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
        #fork_from=config.fork_from,
        # **config.kwargs,
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


def write_example_to_disk(engine: Engine, evaluators: dict[str, Engine], config: brainnet.config.ResultsParameters):
    engine.add_event_handler(
        config.save_example_on,
        event_handlers.write_example,
        evaluators=evaluators,
        config=config,
    )


def add_metric_to_engine(engine):
    metric = CriterionAggregator()
    metric.attach(engine, "loss")


def add_evaluation_event(
    engine: Engine,
    criterion: Criterion,
    model: BrainNet,
    synth: brainsynth.Synthesizer,
    dataloader,
    evaluate_on: CallableEventWithFilter,
    logger: Callable,
    epoch_length: int,
    enable_amp: bool,
) -> Engine:

    metric = CriterionAggregator()

    eval_step = EvaluationStep(
        synth,
        model,
        criterion,
        enable_amp,
    )
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
        epoch_length=epoch_length,
        logger=logger,
    )

    return evaluator


class SupervisedStep:
    def __init__(
        self,
        synthesizer: None | brainsynth.Synthesizer,
        model: brainnet.BrainNet,
        criterion: brainnet.Criterion,
        optimizer: None | torch.optim.Optimizer = None,
    ) -> None:
        self.synthesizer = synthesizer
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = self.model.device

        # Get empty TemplateSurfaces. We update the vertices at each iteration
        self.surface_template = dict(
            y_pred=self.get_placeholder_surface_templates(),
            y_true=self.get_placeholder_surface_templates(),
        )

    def get_placeholder_surface_templates(self):
        """Initialize placeholder objects for the predicted and target
        surfaces. The vertices are updated on each iteration.
        """
        surface_names = ("white", "pial")

        module = [i for i in self.model.heads.values() if isinstance(i, SurfaceModule)]
        if len(module) == 0:
            return None

        assert len(module) == 1
        module = module[0]

        topology = module.get_prediction_topology()
        topology = dict(lh=topology, rh=copy.deepcopy(topology))
        topology["rh"].reverse_face_orientation()

        return {
            h: {
                s: TemplateSurfaces(torch.zeros(t.n_vertices, 3, device=self.device), t)
                for s in surface_names
            }
            for h, t in topology.items()
        }

    def update_surface_template(self, template, data):
        """Insert vertices data from `data` into template and replace `data`
        with the template.
        """
        for h, surfaces in data.items():
            for s in surfaces:
                template[h][s].vertices = surfaces[s]
                data[h][s] = template[h][s]

    def prepare_batch(self, batch):
        """Run data augmentation/synthesis on the batch as returned by the
        dataloader.
        """
        if self.synthesizer is None:
            # assume synthesizer was applied when loading the data
            return batch
        else:
            images, surfaces, init_verts = batch

            # REMOVE BATCH DIMENSION
            for k, v in images.items():
                images[k] = v.squeeze(0)
            for k, v in surfaces.items():
                for kk, vv in v.items():
                    surfaces[k][kk] = vv.squeeze(0)
            for k, v in init_verts.items():
                init_verts[k] = v.squeeze(0)

            with torch.no_grad():
                y_true = self.synthesizer(images, surfaces, init_verts, unpack=False)

            image = y_true.pop("image")
            init_verts = y_true.pop("initial_vertices")

            # ADD BATCH DIMENSION
            image = image.unsqueeze(0)
            for k, v in y_true.items():
                if k == "surface":
                    for k2, v2 in v.items():
                        for k3, v3 in v2.items():
                            y_true[k][k2][k3] = v3.unsqueeze(0)
                else:
                    y_true[k] = v.unsqueeze(0)
            for k, v in init_verts.items():
                init_verts[k] = v.unsqueeze(0)

            return image, y_true, init_verts

    def prepare_loss(self, y_pred, y_true):
        if (k := "surface") in y_pred:
            self.update_surface_template(self.surface_template["y_pred"], y_pred[k])
            self.update_surface_template(self.surface_template["y_true"], y_true[k])

            self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

    def compute_loss(self, y_pred, y_true):
        self.prepare_loss(y_pred, y_true)
        raw = self.criterion(y_pred, y_true)
        return dict(
            raw=self.criterion(y_pred, y_true),
            weighted=self.criterion.apply_weights(raw),
        )


class SupervisedTrainingStep(SupervisedStep):
    def __init__(
        self, synthesizer, model, criterion, optimizer, enable_amp: bool = False
    ) -> None:
        super().__init__(synthesizer, model, criterion, optimizer)
        assert self.optimizer is not None, "Optimizer must be provided for training"
        self.enable_amp = enable_amp

    def __call__(self, engine, batch) -> tuple:
        # Reset gradients in optimizer. Otherwise gradients would
        # accumulate across multiple passes (whenever .backward is
        # called)
        self.optimizer.zero_grad()

        image, y_true, init_verts = self.prepare_batch(batch)

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        self.model.train()
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            y_pred = self.model(
                image,
                init_verts,  # head_kwargs=engine.state.head_runtime_kwargs
            )
            loss = self.compute_loss(y_pred, y_true)
            total_loss = recursive_dict_sum(loss["weighted"])

        # exit if loss diverges
        if total_loss > 1e6 or torch.isnan(total_loss):
            raise RuntimeError(f"Loss diverged (loss = {total_loss})")

        total_loss.backward()  # backpropagate loss
        self.optimizer.step()  # update parameters

        loss = recursive_itemize(loss)

        # these are stored in engine.state.output
        return loss, image, y_pred, y_true


class EvaluationStep(SupervisedStep):
    def __init__(self, synthesizer, model, criterion, enable_amp: bool = False):
        super().__init__(synthesizer, model, criterion, optimizer=None)
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        image, y_true, init_verts = self.prepare_batch(batch)

        self.model.eval()
        with torch.inference_mode():
            with torch.autocast(self.device.type, enabled=self.enable_amp):
                y_pred = self.model(
                    image, init_verts  # , head_kwargs=engine.state.head_runtime_kwargs
                )
                loss = self.compute_loss(y_pred, y_true)

        # we don't need the weighted loss
        del loss["weighted"]
        loss = recursive_itemize(loss)

        return loss, image, y_pred, y_true


# def initialize(config):

#     if hasattr(config, "lr_scheduler") and config.lr_scheduler is not None:
#         lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.model)(
#             optimizer, **vars(config.lr_scheduler.kwargs)
#         )
#     else:
#         lr_scheduler = None

#     return dataloader, synthesizer, model, optimizer, criterion, lr_scheduler


def train(args):
    # args.config

    train_setup_file = args.config  # "brainnet.config.surface_model.main"

    print("Setting up training...")

    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")

    sep_line = 79 * "="

    # Overwrite args from command line if provided
    if args.load_checkpoint is not None:
        train_setup.train_params.load_checkpoint = args.load_checkpoint
    if args.max_epochs is not None:
        train_setup.train_params.max_epochs = args.max_epochs
    if args.no_wandb:
        train_setup.wandb.enable = False

    criterion = brainnet.initializers.init_criterion(train_setup.criterion)
    dataloader = brainnet.initializers.init_dataloader(
        train_setup.dataset, train_setup.dataloader
    )
    model = brainnet.initializers.init_model(train_setup.model)
    optimizer = brainnet.initializers.init_optimizer(train_setup.optimizer, model)
    synth = brainnet.initializers.init_synthesizer(train_setup.synthesizer)

    # =============================================================================
    # TRAINING
    # =============================================================================

    train_step = SupervisedTrainingStep(
        synth["train"],
        model,
        criterion["train"],
        optimizer,
        enable_amp=train_setup.train_params.enable_amp,
    )
    trainer = Engine(train_step)

    # The order in which the events are added to the engine is important!

    # Aggregate average loss over epoch
    add_metric_to_engine(trainer)
    add_terminal_logger(trainer)

    # Add evaluation
    kwargs = dict(
        engine=trainer,
        model=model,
        evaluate_on=train_setup.train_params.evaluate_on,
        epoch_length=train_setup.train_params.epoch_length_val,
        enable_amp=train_setup.train_params.enable_amp,
    )
    evaluators = dict(
        # train = add_evaluation_event(
        #     criterion=criterion["validation"],  # NOTE here we use the validation criterion!
        #     synth=synth["train"],
        #     dataloader=dataloader["train"],
        #     logger=event_handlers.MetricLogger(key="loss", name="train"),
        #     **kwargs,
        # ),
        validation=add_evaluation_event(
            criterion=criterion["validation"],
            synth=synth["validation"],
            dataloader=dataloader["validation"],
            logger=event_handlers.MetricLogger(key="loss", name="validation"),
            **kwargs,
        ),
    )

    add_wandb_logger(trainer, evaluators, train_setup.wandb)

    # Should be triggered after metrics has been computed!
    add_custom_events(trainer, train_setup.train_params.events_trainer)
    for e in evaluators.values():
        add_custom_events(e, train_setup.train_params.events_evaluators)


    # to_save = dict(model=model, optimizer=optimizer, engine=trainer)
    # load_checkpoint(to_save, train_setup)

    # Include this in the checkpoint
    to_save = dict(model=model, optimizer=optimizer, engine=trainer,
                   **{f"criterion[{k}]": v for k,v in criterion.items()})

    add_model_checkpoint(trainer, to_save, train_setup.results)
    write_example_to_disk(trainer, evaluators, train_setup.results)

    load_checkpoint(to_save, train_setup)

    print("Setup completed. Starting training at epoch ...")

    print(sep_line)
    print(f"Config file     {train_setup_file}")
    print(f"Project         {train_setup.project:30s}")
    print(f"Run             {train_setup.run:30s}")
    print(f"Output dir      {train_setup.results.out_dir}")
    print(f"Wandb enabled   {train_setup.wandb.enable}")
    print(sep_line)

    # Start the training
    trainer.run(
        dataloader["train"],
        epoch_length=train_setup.train_params.epoch_length_train,
        max_epochs=train_setup.train_params.max_epochs,
    )


def parse_args(argv):
    description = "Main interface to training a BrainNet model. For convenience, a few parameters are exposed on the command line. Values provided here will overwrite those set in the configuration file."
    parser = argparse.ArgumentParser(
        prog="BrainNetTrainer", description=description,
    )
    parser.add_argument(
        "config", help="Configuration file defining the parameters for training."
    )
    parser.add_argument("--load-checkpoint", default=None, type=int, help="Resume training from this checkpoint.")
    parser.add_argument("--max-epochs", default=None, type=int, help="Terminate training when this number of epochs is reached.")
    parser.add_argument("--no-wandb", action="store_true", default=False, help="Disable logging with wandb.")

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    train(args)

# import torch
# import nibabel as nib
# from brainnet.mesh import surface
# import numpy as np
# import pyvista as pv


# device = torch.device("cuda:0")
# v, f = nib.freesurfer.read_geometry(
#     "/mnt/scratch/personal/jesperdn/results/BrainNetEdgeNet/PointSample/examples/train_800_lh.white.pred"
# )

# pred = surface.TemplateSurfaces(
#     torch.tensor(v.astype(np.float32)).to(device),
#     torch.tensor(f.astype(np.int32)).to(device),
# )

# n = pred.compute_vertex_normals()
# K = pred.compute_laplace_beltrami_operator()
# H = pred.compute_mean_curvature(K)


# v1, f1 = nib.freesurfer.read_geometry(
#     "/mnt/scratch/personal/jesperdn/results/BrainNetEdgeNet/PointSample/examples/train_800_lh.white.true"
# )
# tr = surface.TemplateSurfaces(
#     torch.tensor(v1.astype(np.float32)).to(device),
#     torch.tensor(f1.astype(np.int32)).to(device),
# )

# n1 = pred.compute_vertex_normals()
# K1 = pred.compute_laplace_beltrami_operator()
# H1 = pred.compute_mean_curvature(K1)

# ix = pred.nearest_neighbor(tr)
# ix1 = tr.nearest_neighbor(pred)


# m = pv.make_tri_mesh(v, f)
# m["K"] = K.cpu().numpy()[0]
# m["H"] = H.cpu().numpy()[0]
# m["n"] = n.cpu().numpy()[0]
# m["dK"] = torch.sum((K[0] - K1[0, ix1]) ** 2, dim=-1).cpu().numpy()[0]
# m.save("/home/jesperdn/nobackup/pred.vtk")

# m = pv.make_tri_mesh(v1, f1)
# m["K"] = K1.cpu().numpy()[0]
# m["H"] = H1.cpu().numpy()[0]
# m["n"] = n1.cpu().numpy()[0]
# m["dK"] = torch.sum((K[0, ix] - K1[0]) ** 2, dim=-1).cpu().numpy()[0]
# m.save("/home/jesperdn/nobackup/true.vtk")


# with torch.inference_mode():
#     y_pred = self.model(image, init_verts, head_kwargs=self.head_runtime_kwargs)
# # convert surface predictions to batched surfaces
# if (k := "surface") in y_pred:
#     # insert vertices into template surface
#     self.set_templatesurface(y_pred[k], self.surface_skeletons["y_pred"])
#     # self.set_templatesurface(y_true_out[k], self.surface_skeletons["y_true"])

#     # self.criterion.precompute_for_surface_loss(y_pred[k], y_true[k])
#     self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

# from pathlib import Path
# from brainsynth.dataset import CroppedDataset
# import torch
# from brainnet.mesh import surface
# from brainnet.mesh.topology import get_recursively_subdivided_topology

# base_dir = Path("/mnt/scratch/personal/jesperdn/datasets")
# datasets = ("HCP", "OASIS3")
# kwargs = dict(surface_resolution=6, surface_hemi="both", default_images=[])  # norm
# device = torch.device("cuda:0")

# topology = get_recursively_subdivided_topology(6)
# faces = topology[-1].faces.to(device)
# faces = dict(lh=faces, rh=faces[:, (0, 2, 1)])

# # Individual datasets
# datasets = [
#     CroppedDataset(
#         base_dir / ds,
#         optional_images=None,
#         dataset_id=ds,
#         return_dataset_id=False,
#         **kwargs,
#     )
#     for ds in datasets
# ]
# dataset = torch.utils.data.ConcatDataset(datasets)

# n_subjects = len(dataset)

# stats = {}
# for hemi in ("lh", "rh"):
#     print(hemi)
#     stats[hemi] = {}
#     for ss in ("white", "pial"):
#         print(ss)
#         stats[hemi][ss] = {}
#         H = torch.zeros((topology[-1].n_vertices, n_subjects), device=device)
#         for i in range(n_subjects):
#             if i % 100 == 0:
#                 print(i)
#             _, surf, init, info = dataset[i]
#             s = surface.TemplateSurfaces(surf[hemi][ss].to(device), faces[hemi])
#             K = s.compute_laplace_beltrami_operator()
#             H[:, i] = s.compute_mean_curvature(K).squeeze()

#         stats[hemi][ss]["H_mean"] = H.mean(1).cpu()
#         stats[hemi][ss]["H_std"] = H.std(1).cpu()
#         for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99):
#             stats[hemi][ss][f"H_{q:.2f}"] = H.quantile(q, dim=1).cpu()

# torch.save(stats, "/home/jesperdn/nobackup/prior_curvature_H.pt")

# _, surf, init, info = dataset[0]

# for hemi in ("lh", "rh"):
#     for ss in ("white", "pial"):
#         m = pv.make_tri_mesh(surf[hemi][ss].numpy(), faces[hemi].cpu().numpy())
#         for st in stats[hemi][ss]:
#             m[st] = stats[hemi][ss][st].numpy()
#         m.save(f"/home/jesperdn/nobackup/H_prior_{hemi}_{ss}.vtk")
