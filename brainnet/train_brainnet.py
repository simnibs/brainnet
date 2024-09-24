import copy
import importlib
import sys
import torch

from ignite.engine import Engine

import brainsynth

import brainnet.config
import brainnet.train_utilities

from brainnet import event_handlers
import brainnet.initializers
from brainnet.mesh.surface import TemplateSurfaces
from brainnet.modules.head import SurfaceModule
from brainnet.utilities import recursive_dict_sum, recursive_itemize


# points = np.concatenate((w.vertices[0].cpu().numpy(), p.vertices[0].cpu().numpy()))
# cells = np.concatenate(
#     (
#         np.full((w.topology.n_vertices, 1), 2),
#         np.ascontiguousarray(np.arange(len(points)).reshape(2, -1).T),
#     ),
#     axis=1,
# )
# lines = pv.PolyData(points, cells)



class SupervisedStep:
    def __init__(
        self,
        synthesizer: None | brainsynth.Synthesizer,
        model: brainnet.BrainNet,
        criterion: brainnet.Criterion,
    ) -> None:
        self.synthesizer = synthesizer
        self.model = model
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
        super().__init__(synthesizer, model, criterion)
        self.optimizer = optimizer
        self.enable_amp = enable_amp

    def __call__(self, engine, batch) -> tuple:
        # Reset gradients in optimizer. Otherwise gradients would
        # accumulate across multiple passes (whenever .backward is
        # called)
        self.optimizer.zero_grad()

        image, y_true, init_verts = self.prepare_batch(batch)

        # Only wrap forward pass and loss computation. Backward uses the same11
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
        super().__init__(synthesizer, model, criterion)
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
    brainnet.train_utilities.add_metric_to_engine(trainer)
    brainnet.train_utilities.add_terminal_logger(trainer)

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
        validation=brainnet.train_utilities.add_evaluation_event(
            criterion=criterion["validation"],
            synth=synth["validation"],
            dataloader=dataloader["validation"],
            logger=event_handlers.MetricLogger(key="loss", name="validation"),
            **kwargs,
        ),
    )

    brainnet.train_utilities.add_wandb_logger(trainer, evaluators, train_setup.wandb)

    # Should be triggered after metrics has been computed!
    brainnet.train_utilities.add_custom_events(trainer, train_setup.train_params.events_trainer)
    for e in evaluators.values():
        brainnet.train_utilities.add_custom_events(e, train_setup.train_params.events_evaluators)


    # to_save = dict(model=model, optimizer=optimizer, engine=trainer)
    # load_checkpoint(to_save, train_setup)

    # Include this in the checkpoint
    to_save = dict(model=model, optimizer=optimizer, engine=trainer,
                   **{f"criterion[{k}]": v for k,v in criterion.items()})

    brainnet.train_utilities.add_model_checkpoint(trainer, to_save, train_setup.results)
    brainnet.train_utilities.write_example_to_disk(trainer, evaluators, train_setup.results)

    brainnet.train_utilities.load_checkpoint(to_save, train_setup)

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

if __name__ == "__main__":
    args = brainnet.train_utilities.parse_args(sys.argv)
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
