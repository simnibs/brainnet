import argparse
import copy
import functools
from datetime import datetime
import sys

import torch

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from brainnet import event_handlers

import wandb

import brainsynth
from brainsynth.config.utilities import load_config, recursive_namespace_to_dict
from brainsynth.dataset import get_dataloader_concatenated_and_split

from brainnet.mesh.surface import TemplateSurfaces
from brainnet.modules.brainnet import BrainNet
from brainnet.modules.head import SurfaceModule
from brainnet.modules.criterion import Criterion
from brainnet.utilities import recursive_dict_sum

fmt_epoch = lambda epoch: f"{epoch:05d}"
fmt_state = lambda epoch: f"state_{fmt_epoch(epoch)}.pt"

import nibabel as nib

# import warnings
# warnings.simplefilter("error")

points = np.concatenate((w.vertices[0].cpu().numpy(), p.vertices[0].cpu().numpy()))
cells = np.concatenate(
    (
        np.full((w.topology.n_vertices, 1), 2),
        np.ascontiguousarray(np.arange(len(points)).reshape(2, -1).T),
    ),
    axis=1,
)
lines = pv.PolyData(points, cells)


# or
def recursive_itemize(d, out=None):
    """Recursively call .item() on values."""
    if out is None:
        out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            recursive_itemize(v, out)
        else:
            out[k] = v.item()
    return out


class SupervisedStep:
    def __init__(self, synthesizer, model, criterion, optimizer=None) -> None:
        self.synthesizer = synthesizer
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

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
            with torch.no_grad():
                y_true = self.synthesizer(images, surfaces, init_verts, unpack=False)
            image = y_true.pop("image")
            init_verts = y_true.pop("initial_vertices")
            return image, y_true, init_verts

    def prepare_loss(self, y_pred, y_true):
        if (k := "surface") in y_pred:
            self.update_surface_template(self.surface_template["y_pred"], y_pred[k])
            self.update_surface_template(self.surface_template["y_true"], y_true[k])

            self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

    def compute_loss(self, y_pred, y_true):
        self.prepare_loss(y_pred, y_true)
        loss = dict(raw=self.criterion(y_pred, y_true))
        loss["weighted"] = self.criterion.apply_weights(loss)
        return loss


class SupervisedTrainingStep(SupervisedStep):
    def __init__(self, synthesizer, model, criterion, optimizer) -> None:
        super().__init__(synthesizer, model, criterion, optimizer)
        assert self.optimizer is not None, "Optimizer must be provided for training"

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        # Reset gradients in optimizer. Otherwise gradients would
        # accumulate across multiple passes (whenever .backward is
        # called)
        self.optimizer.zero_grad()

        image, y_true, init_verts = self.prepare_batch(batch)

        # Predict
        y_pred = self.model(
            image, init_verts, head_kwargs=engine.state.head_runtime_kwargs
        )

        loss = self.compute_loss(y_pred, y_true)
        total_loss = recursive_dict_sum(loss["weighted"])
        total_loss.backward()

        # # exit if loss diverges
        # if wloss_sum > 1e6 or torch.isnan(wloss_sum):
        #     sys.exit()

        # Update parameters (i.e., gradients)
        self.optimizer.step()

        loss = recursive_itemize(loss)

        # stored in engine.state.output
        return loss, y_pred, y_true


class ValidationStep(SupervisedStep):
    def __init__(self, synthesizer, model, criterion):
        super().__init__(synthesizer, model, criterion, optimizer=None)

    def __call__(self, engine, batch):
        self.model.eval()

        image, y_true, init_verts = self.prepare_batch(batch)

        with torch.inference_mode():
            y_pred = self.model(
                image, init_verts, head_kwargs=engine.state.head_runtime_kwargs
            )

            loss = self.compute_loss(y_pred, y_true)

        return loss


def initialize_dataloaders(ds_config: dict, dl_config: dict):
    subsets = ("train", "validation")
    # the latter dict has presedence
    return {
        s: setup_dataloader(
            DatasetConfig(**config[s]).dataset_kwargs,
            dl_config
        )
        for s in subsets
    }

def initialize_synthesizer(config: dict):
    device = torch.device(config.pop("device"))
    synth = Synthesizer(SynthesizerConfig(**config, device=device))
    synth.to(device)
    return synth



def initialize_wandb(engine, config):


def initialize(config):



    if hasattr(config, "lr_scheduler") and config.lr_scheduler is not None:
        lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.model)(
            optimizer, **vars(config.lr_scheduler.kwargs)
        )
    else:
        lr_scheduler = None

    return dataloader, synthesizer, model, optimizer, criterion, lr_scheduler

def initialize_optimizer(model, config):

    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Number of trainable parameters: {n_parameters}")

    if config.lr_parameter_groups is not None:
        lr_pg = config.lr_parameter_groups
        parameters = []

        # body network
        d = dict(params=model.body.parameters())
        if hasattr(lr_pg, "body"):
            d["lr"] = lr_pg.body
        parameters.append(d)

        # Task networks
        if hasattr(lr_pg, "heads"):
            for k, v in model.heads.items():
                d = dict(params=v.parameters())
                if hasattr(lr_pg.heads, k):
                    d["lr"] = getattr(lr_pg.heads, k)
                parameters.append(d)
        else:
            parameters.append(model.heads.parameters())
    else:
        parameters = model.parameters()

    return getattr(torch.optim, config.name)(parameters, config.kwargs)


def initialize_criterion(config):
    return Criterion(config)


def initialize_model(config):
    # Device is needed as arg for topofit for now...
    device = torch.device(config.device)
    model = BrainNet(config.body, config.heads, device)
    model.to(device)
    return model


def add_state_entries_(trainer, config):
    """Set additional state variables besides the default ones.

    References
    ----------
    https://pytorch.org/ignite/_modules/ignite/engine/events.html#State
    """
    setattr(trainer.state, "evaluation", {})
    setattr(trainer.state, "config", config)
    setattr(
        trainer.state,
        "head_runtime_kwargs",
        {
            k: recursive_namespace_to_dict(v.runtime_kwargs)
            for k, v in vars(config.model.heads).items()
            if hasattr(v, "runtime_kwargs")
        },
    )

    trainer.state._update_attrs()


train_setup = TrainSetup()

validation_interval = 20
log_interval = 20

dataloader = initialize_dataloaders(config.dataset, config.dataloader)
model = initialize_model(config.model)
optimizer = initialize_optimizer(model, config.optimizer)


# Training
train_step = SupervisedTrainingStep(train_synthesizer, model, train_criterion)

trainer = Engine(train_step)
add_state_entries_(trainer, config)



def initialize_events(engine, config):

    # Events from config
    for e in config.events:
        engine.add_event_handler(
            getattr(Events, e.event)(**e.event_filter),
            e.action(**e.action_kwargs)
        )

    # Enable logging with wandb
    if config.enable:
        engine.add_event_handler(Events.STARTED, event_handlers.wandb_init)
        engine.add_event_handler(Events.EPOCH_COMPLETED, event_handlers.wandb_log)
        engine.add_event_handler(Events.COMPLETED, event_handlers.wandb_stop)
    else:
        engine.state.wandb = None


    # TERMINAL logging
    engine.add_event_handler(Events.EPOCH_COMPLETED, event_handlers.TerminalLogger())


# Evaluation
val_step = ValidationStep(val_synthesizer, model, val_criterion)
val_evaluator = Engine(val_step)
log_val_results = functools.partial(
    evalauate_model, evaluator=val_evaluator, loader=val_dataloader
)

train_eval_step = ValidationStep(train_synthesizer, model, val_criterion)
train_evaluator = Engine(train_eval_step)
log_train_results = functools.partial(
    evaluate_model, evaluator=train_evaluator, loader=train_dataloader
)

trainer.add_event_handler(
    Events.EPOCH_COMPLETED(every=validation_interval), log_train_results
)
trainer.add_event_handler(
    Events.EPOCH_COMPLETED(every=validation_interval), log_val_results
)


trainer.run(dataloader, ...)

pbar = ProgressBar()
pbar.attach(trainer, output_transform=lambda x: {"loss": x})


@trainer.on(Events.EPOCH_COMPLETED)
def log_epoch_time():
    print(
        f"Epoch {trainer.state.epoch}, Time Taken : {trainer.state.times['EPOCH_COMPLETED']}"
    )


trainer.add_event_handler(Events.EPOCH_COMPLETED, log_epoch_time)

# Save the model after every epoch of val_evaluator is completed
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})


# def on_iteration_completed(engine):
#     iteration = engine.state.iteration
#     epoch = engine.state.epoch
#     loss = engine.state.output
#     print(f"Epoch: {epoch}, Iteration: {iteration}, Loss: {loss}")


# Checkpoint to store n_saved best models wrt score function
model_checkpoint = ModelCheckpoint(
    "ckpt",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(
        trainer
    ),  # helps fetch the trainer's state
)


# Finally, start the training

trainer.run(dataloader, epoch_length=epoch_length, max_epochs=max_epochs)


import torch
import nibabel as nib
from brainnet.mesh import surface
import numpy as np
import pyvista as pv


device = torch.device("cuda:0")
v, f = nib.freesurfer.read_geometry(
    "/mnt/scratch/personal/jesperdn/results/BrainNetEdgeNet/PointSample/examples/train_800_lh.white.pred"
)

pred = surface.TemplateSurfaces(
    torch.tensor(v.astype(np.float32)).to(device),
    torch.tensor(f.astype(np.int32)).to(device),
)

n = pred.compute_vertex_normals()
K = pred.compute_laplace_beltrami_operator()
H = pred.compute_mean_curvature(K)


v1, f1 = nib.freesurfer.read_geometry(
    "/mnt/scratch/personal/jesperdn/results/BrainNetEdgeNet/PointSample/examples/train_800_lh.white.true"
)
tr = surface.TemplateSurfaces(
    torch.tensor(v1.astype(np.float32)).to(device),
    torch.tensor(f1.astype(np.int32)).to(device),
)

n1 = pred.compute_vertex_normals()
K1 = pred.compute_laplace_beltrami_operator()
H1 = pred.compute_mean_curvature(K1)

ix = pred.nearest_neighbor(tr)
ix1 = tr.nearest_neighbor(pred)


m = pv.make_tri_mesh(v, f)
m["K"] = K.cpu().numpy()[0]
m["H"] = H.cpu().numpy()[0]
m["n"] = n.cpu().numpy()[0]
m["dK"] = torch.sum((K[0] - K1[0, ix1]) ** 2, dim=-1).cpu().numpy()[0]
m.save("/home/jesperdn/nobackup/pred.vtk")

m = pv.make_tri_mesh(v1, f1)
m["K"] = K1.cpu().numpy()[0]
m["H"] = H1.cpu().numpy()[0]
m["n"] = n1.cpu().numpy()[0]
m["dK"] = torch.sum((K[0, ix] - K1[0]) ** 2, dim=-1).cpu().numpy()[0]
m.save("/home/jesperdn/nobackup/true.vtk")


with torch.inference_mode():
    y_pred = self.model(image, init_verts, head_kwargs=self.head_runtime_kwargs)
# convert surface predictions to batched surfaces
if (k := "surface") in y_pred:
    # insert vertices into template surface
    self.set_templatesurface(y_pred[k], self.surface_skeletons["y_pred"])
    # self.set_templatesurface(y_true_out[k], self.surface_skeletons["y_true"])

    # self.criterion.precompute_for_surface_loss(y_pred[k], y_true[k])
    self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

from pathlib import Path
from brainsynth.dataset import CroppedDataset
import torch
from brainnet.mesh import surface
from brainnet.mesh.topology import get_recursively_subdivided_topology

base_dir = Path("/mnt/scratch/personal/jesperdn/datasets")
datasets = ("HCP", "OASIS3")
kwargs = dict(surface_resolution=6, surface_hemi="both", default_images=[])  # norm
device = torch.device("cuda:0")

topology = get_recursively_subdivided_topology(6)
faces = topology[-1].faces.to(device)
faces = dict(lh=faces, rh=faces[:, (0, 2, 1)])

# Individual datasets
datasets = [
    CroppedDataset(
        base_dir / ds,
        optional_images=None,
        dataset_id=ds,
        return_dataset_id=False,
        **kwargs,
    )
    for ds in datasets
]
dataset = torch.utils.data.ConcatDataset(datasets)

n_subjects = len(dataset)

stats = {}
for hemi in ("lh", "rh"):
    print(hemi)
    stats[hemi] = {}
    for ss in ("white", "pial"):
        print(ss)
        stats[hemi][ss] = {}
        H = torch.zeros((topology[-1].n_vertices, n_subjects), device=device)
        for i in range(n_subjects):
            if i % 100 == 0:
                print(i)
            _, surf, init, info = dataset[i]
            s = surface.TemplateSurfaces(surf[hemi][ss].to(device), faces[hemi])
            K = s.compute_laplace_beltrami_operator()
            H[:, i] = s.compute_mean_curvature(K).squeeze()

        stats[hemi][ss]["H_mean"] = H.mean(1).cpu()
        stats[hemi][ss]["H_std"] = H.std(1).cpu()
        for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99):
            stats[hemi][ss][f"H_{q:.2f}"] = H.quantile(q, dim=1).cpu()

torch.save(stats, "/home/jesperdn/nobackup/prior_curvature_H.pt")

_, surf, init, info = dataset[0]

for hemi in ("lh", "rh"):
    for ss in ("white", "pial"):
        m = pv.make_tri_mesh(surf[hemi][ss].numpy(), faces[hemi].cpu().numpy())
        for st in stats[hemi][ss]:
            m[st] = stats[hemi][ss][st].numpy()
        m.save(f"/home/jesperdn/nobackup/H_prior_{hemi}_{ss}.vtk")
