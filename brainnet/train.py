import argparse
import copy
from datetime import datetime
import pprint
import sys
from time import perf_counter

import torch
import wandb

import brainsynth
from brainsynth.config.utilities import load_config, recursive_namespace_to_dict
from brainsynth.dataset import get_dataloader_concatenated_and_split, setup_dataloader

import brainnet
from brainnet.mesh.surface import TemplateSurfaces
from brainnet.modules.brainnet import BrainNet
from brainnet.modules.heads import surface_modules
from brainnet.modules.criterion import Criterion
from brainnet.utilities import recursive_dict_sum

fmt_epoch = lambda epoch: f"{epoch:05d}"
fmt_state = lambda epoch: f"state_{fmt_epoch(epoch)}.pt"

import nibabel as nib


def _config_to_args_dataloader(config):
    tc = config.dataset

    return dict(
        base_dir=tc.dir,
        datasets=tc.train,
        optional_images=recursive_namespace_to_dict(tc.optional_images),
        dataset_kwargs=recursive_namespace_to_dict(tc.kwargs),
        dataloader_kwargs=recursive_namespace_to_dict(tc.dataloader),
        dataset_splits=recursive_namespace_to_dict(tc.split.splits),
        split_rng_seed=tc.split.rng_seed,
    )


"""

def ensure_divisible_shape(shape, strides):
    divisor = strides**len(strides)
    res = shape / divisor
    dif = np.ceil(res) - res
    return (shape + divisor * dif).to(shape.dtype)

"""


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


class BrainNetTrainer:
    def __init__(self, config):
        torch.backends.cudnn.benchmark = True  # increases initial time
        torch.backends.cudnn.deterministic = True

        self.config = config

        self.device = torch.device(config.device.model)
        self.synth_device = torch.device(config.device.synthesizer)

        # assert self.device == self.synth_device

        self.epoch = config.epoch
        self.epoch.start = self.epoch.resume_from_checkpoint or 0

        self.latest_checkpoint = None

        print(f"Project: {self.config.PROJECT_NAME}")
        print(f"Run:     {self.config.RUN_NAME}")

        self.results_dir = self.config.results.dir / self.config.PROJECT_NAME

        self.run_dir = self.results_dir / self.config.RUN_NAME
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True)

        self.checkpoint_dir = self.run_dir / "checkpoints"
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

        self.examples_dir = self.run_dir / "examples"
        if not self.examples_dir.exists():
            self.examples_dir.mkdir()

        # VERSION 1

        # self.dataloader = get_dataloader_concatenated_and_split(
        #     **_config_to_args_dataloader(config)
        # )

        # VERSION 2

        self.dataloaders = {}
        for k,v in vars(self.config.dataset.dataloaders).items():
            assert v.dataloader_kwargs.batch_size == 1

            if hasattr(v, "synthesizer_kwargs") and v.synthesizer_kwargs is not None:
                synthesizer = brainsynth.Synthesizer(
                    **recursive_namespace_to_dict(v.synthesizer_kwargs)
                )
            else:
                synthesizer = None

            self.dataloaders[k] = setup_dataloader(
                recursive_namespace_to_dict(v.subjects),
                synthesizer,
                recursive_namespace_to_dict(v.dataset_kwargs),
                recursive_namespace_to_dict(v.dataloader_kwargs),
            )


        print("Dataloader Subsets")
        for k, v in self.dataloaders.items():
            print(f"# samples in {k:15s}      {len(v):5d}")

        if hasattr(self.config, "synthesizer"):
            self.synthesizer = brainsynth.Synthesizer(
                self.config.synthesizer.config,
                # ensure_divisible_by=divisor,
                device=self.synth_device,
            )
            self.synthesizer.to(self.synth_device)
        else:
            self.synthesizer = None

        # Logging
        if hasattr(self.config, "wandb") and self.config.wandb.enable:
            self.wandb = wandb
            self.wandb_dir = self.results_dir
            if not self.wandb_dir.exists():
                self.wandb_dir.mkdir(parents=True)
        else:
            self.wandb = None

        # Device is needed as arg for topofit for now...
        self.model = BrainNet(config.model.body, config.model.heads, self.device)
        self.model.to(self.device)

        # Task specific kwargs
        self.head_runtime_kwargs = {
            k: recursive_namespace_to_dict(v.runtime_kwargs)
            for k, v in vars(self.config.model.heads).items()
            if hasattr(v, "runtime_kwargs")
        }

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters: {n_parameters}")

        if hasattr(self.config.optimizer, "lr_parameter_groups"):
            lr_pg = self.config.optimizer.lr_parameter_groups
            parameters = []

            # body network
            d = dict(params=self.model.body.parameters())
            if hasattr(lr_pg, "body"):
                d["lr"] = lr_pg.body
            parameters.append(d)

            # Task networks
            if hasattr(lr_pg, "heads"):
                for k, v in self.model.heads.items():
                    d = dict(params=v.parameters())
                    if hasattr(lr_pg.heads, k):
                        d["lr"] = getattr(lr_pg.heads, k)
                    parameters.append(d)
            else:
                parameters.append(self.model.heads.parameters())
        else:
            parameters = self.model.parameters()

        self.initialize_optimizer(parameters)

        # load model and optimizer parameters
        self.load_checkpoint(self.epoch.resume_from_checkpoint)

        # Freeze parameters of feature extractor
        # self.model.body.requires_grad_(False)

        # only update pial surface parameters...
        # for param in self.model.body.parameters():
        #     param.requires_grad = False
        # for param in self.model.heads.surface.unet_deform.parameters():
        #     param.requires_grad = False


        if hasattr(self.config.optimizer, "lr_factor") and self.config.optimizer.lr_factor != 1.0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.config.optimizer.lr_factor

        self.criterion = Criterion(config.loss)

        if hasattr(config, "lr_scheduler") and config.lr_scheduler is not None:
            match config.lr_scheduler.model:
                case "OneCycleLR":
                    kw = dict(max_lr=[p["lr"] for p in self.optimizer.param_groups])
                case "CyclicLR":
                    kw = dict(
                        base_lr=[p["lr"] for p in self.optimizer.param_groups],
                    )
                    kw["max_lr"] = [
                        config.lr_scheduler.kwargs.max_lr_factor * lr
                        for lr in kw["base_lr"]
                    ]
                    del config.lr_scheduler.kwargs.max_lr_factor
                case "LinearLR":
                    pass
                case _:
                    raise ValueError("Invalid LR scheduler")

            self.lr_scheduler = getattr(
                torch.optim.lr_scheduler, config.lr_scheduler.model
            )(self.optimizer, **kw, **vars(config.lr_scheduler.kwargs))
        else:
            self.lr_scheduler = None

        # self.lr_schedulers = torch.optim.lr_scheduler.ChainedScheduler(
        #     [
        #         getattr(torch.optim.lr_scheduler, k)(self.optimizer, **v)
        #         for k, v in config["lr_schedulers"].items()
        #     ]
        # )

        # for g in self.optimizer.param_groups:
        #     g["lr"] = 1e-8

        # Get empty TemplateSurfaces. We update the vertices at each iteration
        self.surface_skeletons = dict(
            y_pred=self.get_surface_skeletons(),
            y_true=self.get_surface_skeletons(),
        )

        # SET MEDIAL WALL WEIGHTS...

        # h = next(iter(self.surface_skeletons["y_true"]))
        # medialwall = torch.load(brainnet.resources_dir / "medial-wall.pt").to(device=self.device)
        # weights = torch.ones(medialwall.size(), device=self.device)
        # weights[medialwall] = 0.25
        # weights = weights[:self.surface_skeletons["y_true"][h]["white"].topology.n_vertices]
        # weights = weights[None]
        # for h in self.surface_skeletons["y_true"]:
        #     self.surface_skeletons["y_true"][h]["white"].vertex_data["weights"] = weights
        #     self.surface_skeletons["y_true"][h]["pial"].vertex_data["weights"] = weights


    def initialize_optimizer(self, parameters=None):
        parameters = parameters or self.model.parameters()
        self.optimizer = getattr(torch.optim, self.config.optimizer.name)(
            parameters,
            **recursive_namespace_to_dict(self.config.optimizer.kwargs),
        )

    def _get_lr(self):
        return {
            f"group_{i:02d}": param_group["lr"]
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

    def has_converged(self):
        return self._get_lr() <= self.config.convergence.minimum_lr

    def get_hyper_params(self):
        return dict(
            LR=self._get_lr(),
        )

    def to_device(self, tensors, device=None, out=None):
        device = device or self.device
        out = out or {}
        match tensors:
            case torch.Tensor():
                return tensors.to(self.device, non_blocking=True)
            case dict():
                for k, v in tensors.items():
                    tensors[k] = self.to_device(v, device)
        return tensors

    def train(self):
        """Train a model until 'convergence' or maximum number of epochs is
        reached.
        """
        fmt = "%B %d, %Y at %H:%M:%S"
        start = datetime.now()
        print(f"Training started on {start.strftime(fmt)}")

        self.model.train()
        self.wandb_init()

        print("Using hyperparameters", self.get_hyper_params())

        # train_sampler = iter(self.dataloaders["train"])

        for n in range(self.epoch.start + 1, self.epoch.max_allowed_epochs + 1):
            if hasattr(self.config.reinitialize_loss, str(n-1)):
                self.reinitialize_criterion(n-1)

            epoch = Epoch(n, self.criterion.active_losses)


            hyper_params = self.get_hyper_params()

            TAG = {}
            TAG_count = {}

            # torch.cuda.memory._record_memory_history()

            t_init = perf_counter()

            # a = perf_counter()
            # c = perf_counter()
            # for i in range(10):
            #     x = self.dataloaders["train"].dataset.datasets[0][i]
            #     b = perf_counter()
            #     print(b-a)
            #     a = b
            # print(perf_counter() - c)


            # for step, (image, y_true, init_verts) in next(train_sampler):

            # t = 0.5
            # n = 40
            # c = perf_counter()
            # a = perf_counter()
            # for step, (image, y_true, init_verts) in enumerate(self.dataloaders["train"]):
            #     time.sleep(t)
            #     b = perf_counter()
            #     print(f"{step:2d} : {b-a}")
            #     a = b
            #     if step==n:
            #         break
            # total = perf_counter()-c
            # print(total)
            # print(total - t*n)

            # for step, (ds_id, images, surfaces, temp_verts, info) in enumerate(
            #     self.dataloaders["train"]
            # ):

            # for step, (image, y_true, init_verts) in enumerate(self.dataloaders["train"]):

            for step, (images, surfaces, initial_vertices, info) in enumerate(
                self.dataloaders["train"]
            ):
            # for step, (image, y_true, init_verts) in next(train_sampler):

                if step == self.epoch.steps_per_epoch:
                    break


                if self.synthesizer is not None:
                #     image, y_true, init_verts = self.synthesizer(image, y_true, init_verts)
                    # ds_id is a tuple of length 1
                    image, y_true, init_verts = self.apply_synthesizer(
                        images, surfaces, initial_vertices, info,
                    )

                # decouple white and pial surfaces in y_true slightly
                if self.config.surface_decoupling.decouple:
                    amount = self.config.surface.decoupling.decouple_amount
                    if (k := "surface") in y_true:
                        for h in y_true[k]:
                            white = y_true[k][h]["white"]
                            pial = y_true[k][h]["pial"]

                            zero = torch.norm(pial.vertices - white.vertices, dim=2) < 1e-3
                            normals = pial.compute_vertex_normals()

                            y_true[k][h]["pial"].vertices[zero] += amount * normals[zero]

                # image, y_true, init_verts = self.prepare_data(image, y_true, init_verts)

                if self.config.estimate_ITA:
                    # Hack to enable backpropping through the model multiple times using
                    # the same graph even though the model has technically been updated.

                    # We want to update the model wrt. the loss from one task, compute
                    # losses for the other heads, reset the model (and optimizer), and
                    # repeat for the other heads. Even if the model and optimizer states
                    # are reset, torch will still complain on the second backward call
                    # because the model has changed (although reset!) and so not allow a
                    # backprop again.

                    # When hooks are set for saved tensors (in autograd), torch will not
                    # check tensor versions and hence the second backward call will
                    # succeed. This is a hack and potentially dangerous but I checked that
                    # it gives similar results to recomputing the loss (which has not
                    # changed because the model has been reset) on every task iteration.

                    # Reason for .detach()
                    # https://github.com/pytorch/pytorch/issues/115255

                    pack_hook = lambda x: x.detach()
                    unpack_hook = lambda x: x

                    with torch.autograd.graph.saved_tensors_hooks(
                        pack_hook, unpack_hook
                    ):
                        loss, wloss, y_pred = self.step(image, y_true, init_verts)
                else:
                    # in case of gradient issues, try using
                    #
                    #   with torch.autograd.set_detect_anomaly(True):
                    loss, wloss, y_pred = self.step(image, y_true, init_verts)

                if step == 0 and n % 20 == 0:
                    self.write_results("train", n, image, y_pred, y_true, init_verts)

                # Free some memory before backward pass (the segmentation uses
                # a lot of memory)
                # y_pred is not strictly needed here but we want to save some
                del y_pred

                wloss_sum = recursive_dict_sum(wloss)
                # exit if loss diverges
                if wloss_sum > 1e6 or torch.isnan(wloss_sum):
                    sys.exit()

                if self.config.estimate_ITA:
                    affinity = self.estimate_ITA(image, y_true, init_verts, wloss)

                    for k, v in affinity.items():
                        try:
                            TAG[k] += v
                            TAG_count[k] += 1
                        except KeyError:
                            TAG[k] = v
                            TAG_count[k] = 1

                # Compute and accumulate gradients. backward() frees
                # intermediate values of the graph (e.g., activations)
                wloss_sum.backward()

                # Update parameters (i.e., gradients)
                self.optimizer.step()

                # Reset gradients in optimizer. Otherwise gradients would
                # accumulate across multiple passes (whenever .backward is
                # called)
                self.optimizer.zero_grad()

                #torch.cuda.empty_cache()

                # log the loss
                epoch.loss_update(loss, wloss)

                #self.hook_on_step_end(step)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            timing = dict()
            timing["total (clock)"] = perf_counter() - t_init

            # torch.cuda.memory._dump_snapshot("/home/jesperdn/nobackup/my_snapshot.pickle")

            epoch.loss_normalize()
            epoch.loss_sum()

            epoch.print()

            val_loss = self.validate(epoch.epoch)

            if self.config.estimate_ITA:
                hyper_params["ITA"] = {
                    " -> ".join(k): v / TAG_count[k] for k, v in TAG.items()
                }

            self._wandb_log(epoch.epoch, epoch.loss, val_loss, hyper_params, timing)

            self.hook_on_epoch_end(n)

            self.save_checkpoint(n)


        end = datetime.now()

        print(f"Training ended on   {end.strftime(fmt)}")
        print(f"Time elapsed        {end-start}")

        self.wandb_finish()

    def reinitialize_criterion(self, n):
        loss_config_file = brainnet.config_dir / getattr(self.config.reinitialize_loss, str(n))
        config = load_config(loss_config_file)
        print(f"Reinitializing criterion after epoch {n:5d}")
        print(f"Loading {loss_config_file}")
        print()

        # pprint.pprint(recursive_namespace_to_dict(config))
        self.criterion = Criterion(config)

        print(f"Currently active losses")
        pprint.pprint(self.criterion.active_losses)
        print()

        print(f"Current loss weights")
        pprint.pprint(self.criterion.loss_weights)
        print()

    def estimate_ITA(self, image, y_true, init_verts, wloss):
        """Inter-task affinity (ITA) estimation.

        NOTE THE GRAPH IS NOT CLEARED!!!


        affinity: dict
            (from_task, to_task): value. Positive means that from_task helps
            decrease the loss of to_task (vice versa for negative).


        References
        ----------
        Fifty (2021). Efficiently Identifying Task Groupings for Multi-Task
            Learning. https://arxiv.org/abs/2109.04617

        https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html


        from brainnet.train import *
        config = load_config(
            "/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_foundation_t1w.yaml"
        )
        self = BrainNetTrainer(config)
        image,y_true,init_vert = self.dataloaders["train"].dataset.datasets[0][0]
        image,y_true,init_vert = self.prepare_data(image, y_true,init_vert)

        y_pred = self.model(image, init_vert)


        from brainnet.train import *
        config = load_config("/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_foundation.yaml")
        self = BrainNetTrainer(config)
        ds_id, images, surfaces, temp_verts, info = next(iter(self.dataloader["train"]))
        self.optimizer.zero_grad()
        image, y_true, init_verts = self.apply_synthesizer(images, surfaces, temp_verts, info, ds_id[0])

        loss, wloss, y_pred, affinity = self.step_with_TAG(image, y_true, init_verts)

        with torch.inference_mode():
            self.step(image, y_true, init_verts)

        from brainnet.train import BrainNetTrainer, load_config
        config = load_config("/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_foundation.yaml")
        self = BrainNetTrainer(config)
        ds_id, images, surfaces, temp_verts, info = next(iter(self.dataloader["train"]))
        self.optimizer.zero_grad()
        image, y_true, init_verts = self.apply_synthesizer(images, surfaces, temp_verts, info, ds_id[0])

        a1 = self.step_with_TAG(image, y_true, init_verts)
        a2 = self.step_with_TAG(image, y_true, init_verts)
        a3 = self.step_with_TAG(image, y_true, init_verts)
        a4 = self.step_with_TAG(image, y_true, init_verts)

        a = time.perf_counter()
        _ = self.step(image, y_true, init_verts)
        print(time.perf_counter()-a)


        a = time.perf_counter()
        _ = self.step_with_TAG(image, y_true, init_verts)
        print(time.perf_counter()-a)

        for k,v in self.model.named_parameters():
            print(torch.a/mnt/xnat/INN/arc001/l2028bs( (v - model_state[k]) / v).max())

        for k,v in self.model.named_parameters():
            print(f"{k:80s} {v.shape}")
        """
        # state_dict() returns a shallow copy!
        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())

        n_losses_minus_one = len(self.criterion.losses) - 1
        assert n_losses_minus_one >= 0

        # Update model using "master" task
        affinity = {}
        for mt, mt_loss in wloss.items():

            # Update the shared parameters (as well as the parameters for
            # `task` although we do not investigate the effect on its own task
            # loss)
            total_loss = sum(mt_loss.values())
            total_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # torch.cuda.empty_cache()

            # Updated losses
            with torch.inference_mode():
                _, wloss_updated, _ = self.step(image, y_true, init_verts)

                # Compute affinity between "master" and "slave" heads
                for st, st_loss in wloss_updated.items():
                    if st != mt:
                        # we may have several losses per task
                        ratio = sum(st_loss.values()) / sum(wloss[st].values())
                        affinity[(mt, st)] = 1 - ratio.item()

            # Reset model and optimizer

            # load_state_dict() copies tensor data inplace from state_dict to
            # model, however, with `assign` we copy the tensors anew (along
            # with metadata, e.g., _version attribute). We do this to do a more
            # complete reset of the model.

            # However, since the optimizer holds a *reference* to the parameters in
            # the model, we need to reinitialize the optimizer with the "new"
            # parameters.
            #
            # see https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html

            # if i < n_losses_minus_one:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optim_state)
            # else:
            #     self.model.load_state_dict(model_state, assign=True)
            #     self.initialize_optimizer()
            #     self.optimizer.load_state_dict(optim_state)

        return affinity

    def write_results(self, prefix, n, image, y_pred, y_true, init_verts):
        nib.Nifti1Image(
            image.squeeze().to(torch.float).detach().cpu().numpy(), torch.eye(4).numpy()
        ).to_filename(self.examples_dir / f"{prefix}_{n:04d}_norm.nii.gz")

        if "segmentation" in y_pred:
            nib.Nifti1Image(
                y_pred["segmentation"][0].argmax(0).to(torch.int16).cpu().numpy(),
                torch.eye(4).numpy(),
            ).to_filename(self.examples_dir / f"{prefix}_{n:04d}_seg.nii.gz")

        if "surface" in y_pred:
            for hemi, s in y_pred["surface"].items():
                for surface, ss in s.items():
                    nib.freesurfer.write_geometry(
                        self.examples_dir / f"{prefix}_{n:04d}_{hemi}.{surface}.true",
                        y_true["surface"][hemi][surface]
                        .vertices[0]
                        .to(torch.float)
                        .detach()
                        .cpu()
                        .numpy(),
                        y_true["surface"][hemi][surface]
                        .faces.to(torch.int)
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                    nib.freesurfer.write_geometry(
                        self.examples_dir / f"{prefix}_{n:04d}_{hemi}.{surface}.pred",
                        ss.vertices[0].to(torch.float).detach().cpu().numpy(),
                        ss.faces.to(torch.int).detach().cpu().numpy(),
                    )
                # nib.freesurfer.write_geometry(
                #     self.examples_dir / f"{prefix}_{n:04d}_{hemi}.white.init",
                #     init_verts[hemi][0].to(torch.float).detach().cpu().numpy(),
                #     self.model.heads.surface.topologies[0].faces.to(torch.int).cpu().numpy()
                # )

    def prepare_data(self, image, y_true, initial_vertices):
        image = self.to_device(image)
        y_true = self.to_device(y_true)
        init_vertices = self.to_device(initial_vertices)

        if "surface" in y_true:
            # insert vertices into template surface
            self.set_templatesurface(y_true["surface"], self.surface_skeletons["y_true"])

        return image, y_true, init_vertices

    def apply_synthesizer(
        self, images, surfaces, init_vertices, info, disable_synth=False
    ):

        images = {k:v[0] for k,v in images.items()}
        surfaces = {k:{kk:vv[0] for kk,vv in v.items()} for k,v in surfaces.items()}
        init_vertices = {k:v[0] for k,v in init_vertices.items()}
        info = {k:{kk:vv[0] for kk,vv in v.items()} if isinstance(v, dict) else v[0] for k,v in info.items()}

        if self.synth_device != torch.device("cpu"):
            images = self.to_device(images, self.synth_device)
            surfaces = self.to_device(surfaces, self.synth_device)
            init_vertices = self.to_device(init_vertices, self.synth_device)
            info = self.to_device(info, self.synth_device)

        alternative_synth = ("norm", "T1")

        with torch.no_grad():
            y_true_img, y_true_surf, init_vertices = self.synthesizer(
                images, surfaces, init_vertices, info, disable_synth
            )

            if "synth" not in y_true_img:
                # select a random contrast from the list of alternative images

                # avail = getattr(self.config.dataset.alternative_synth, ds_id)
                avail = alternative_synth

                sel = torch.randint(0, len(avail), (1,), device=self.synthesizer.device)
                y_true_img["synth"] = y_true_img[avail[sel]]

            if self.device != self.synth_device:
                self.to_device(y_true_img)
                self.to_device(y_true_surf)
                self.to_device(init_vertices)

            image = y_true_img.pop("synth")
            y_true = y_true_img

            if len(y_true_surf) > 0:
                # insert vertices into template surface
                self.set_templatesurface(y_true_surf, self.surface_skeletons["y_true"])
            y_true["surface"] = y_true_surf

        image = image[None]
        surfaces = {k:{kk:vv[None] for kk,vv in v.items()} for k,v in surfaces.items()}
        init_vertices = {k:v[None] for k,v in init_vertices.items()}

        return image, y_true, init_vertices


    def step(self, image, y_true, init_verts=None):
        # , model_kwargs=None, criterion_kwargs=None
        # with torch.inference_mode():
        y_pred = self.model(image, init_verts, head_kwargs=self.head_runtime_kwargs)
        # convert surface predictions to batched surfaces
        if (k := "surface") in y_pred:
            # insert vertices into template surface
            self.set_templatesurface(y_pred[k], self.surface_skeletons["y_pred"])

            # self.set_templatesurface(y_true_out[k], self.surface_skeletons["y_true"])

            # self.criterion.precompute_for_surface_loss(y_pred[k], y_true[k])
            self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

        loss = self.criterion(y_pred, y_true)
        wloss = self.criterion.apply_weights(loss)
        return loss, wloss, y_pred

    def _do_save_checkpoint(self, epoch):
        return epoch % self.epoch.save_state_every == 0

    def _do_validation(self, epoch):
        return epoch % self.epoch.validate_every == 0

    def validate(self, epoch):
        if not self._do_validation(epoch):
            return {}

        val_epoch = Epoch(epoch, self.criterion.active_losses)

        self.model.eval()
        with torch.inference_mode():
            # for step, (ds_id, images, surfaces, temp_verts, info) in enumerate(
            #     self.dataloaders["validation"]
            # ):
            # for step, (image, y_true, init_verts) in enumerate(self.dataloaders["validation"]):
            for step, (images, surfaces, initial_vertices, info) in enumerate(
                self.dataloaders["validation"]
            ):
                if step == self.epoch.steps_per_epoch:
                    break

                if self.synthesizer is not None:
                #     image, y_true, init_verts = self.synthesizer(image, y_true, init_verts)
                    # ds_id is a tuple of length 1
                    image, y_true, init_verts = self.apply_synthesizer(
                        images, surfaces, initial_vertices, info, disable_synth=True
                    )

                # image, y_true, init_verts = self.prepare_data(image, y_true, init_verts)

                # same as training step for now but should maybe differ
                loss, wloss, y_pred = self.step(image, y_true, init_verts)

                if step == 0 and epoch % 20 == 0:
                    self.write_results("val", epoch, image, y_pred, y_true, init_verts)

                val_epoch.loss_update(loss, wloss)

            val_epoch.loss_normalize()
            val_epoch.loss_sum()

            print()
            print("VALIDATION:")
            val_epoch.print()
            print()

            self.hook_on_validation_end(val_epoch, val_epoch.loss)
        self.model.train()

        return val_epoch.loss

    def load_checkpoint(self, epoch):
        """Load parameters from a state dict."""
        if epoch is None or epoch == 0:
            return

        name = fmt_state(epoch)
        print(f"Initializing state from {name}")
        state = torch.load(self.checkpoint_dir / name)
        self.model.load_state_dict(state["model_state"], strict=False)
        if self.epoch.load_optimizer_checkpoint:
            self.optimizer.load_state_dict(state["optimizer_state"])

    def save_checkpoint(self, epoch):
        if not self._do_save_checkpoint(epoch):
            return

        torch.save(
            dict(
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
            ),
            self.checkpoint_dir / fmt_state(epoch),
        )
        self.latest_checkpoint = epoch

    def get_surface_skeletons(self):
        surface_names = ("white", "pial")

        module = [i for i in self.model.heads.values() if isinstance(i, surface_modules)]
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

    def set_templatesurface(self, data, surface_skeleton):
        """Replace vertex tensors with TemplateSurfaces objects."""
        for h, surfaces in data.items():
            for s in surfaces:
                surface_skeleton[h][s].vertices = surfaces[s]
                data[h][s] = surface_skeleton[h][s]

    def hook_on_step_end(self, step):
        pass

    def hook_on_epoch_end(self, epoch):
        pass

        # pass

        # epoch_ = str(epoch)

        # # Re-initialize BrainNet using the updated task parameters
        # if epoch_ in self.config.manual_schedulers.heads:
        #     task_config = deepcopy(self.config["heads"])
        #     brainnet.utilities.recursive_dict_update_(
        #         task_config, getattr(self.config.manual_schedulers.heads, epoch_)
        #     )
        #     m = BrainNet(self.config.feature_extractor, task_config)
        #     m.load_state_dict(self.model.state_dict())
        #     self.model = m

        # # Update loss weights
        # if epoch_ in self.config.manual_schedulers.loss_weights:
        #     self.criterion.update_weights(
        #         getattr(self.config.manual_schedulers.loss_weights, epoch_)
        #     )

    def hook_on_validation_end(self, epoch, result):
        pass

    def wandb_init(self):
        if self.wandb is not None:
            self.run = self.wandb.init(
                project=self.config.wandb.project_name,
                name=self.config.wandb.name,
                dir=self.wandb_dir,
                # log the configuration of the run
                config=recursive_namespace_to_dict(self.config),
            )

    def _wandb_log(
        self,
        step: int,
        train_loss: dict,
        val_loss: dict | None = None,
        hyper_params: dict | None = None,
        timing: dict | None = None,
        kwargs: dict | None = None,
    ):
        """

        Parameters
        ----------
        data : dict
            Data to log.
        kwargs :
            Key-word arguments to wandb.log
        """
        if self.wandb is not None:
            kwargs = {} if kwargs is None else kwargs
            data = {"train": train_loss}
            if val_loss:
                data["val"] = val_loss
            if hyper_params is not None:
                data["hyper"] = hyper_params
            if timing is not None:
                data["time"] = timing
            self.wandb.log(data, step=step, **kwargs)

    def wandb_finish(self):
        if self.wandb is not None:
            self.wandb.finish()


class Epoch:
    def __init__(self, epoch, losses) -> None:
        self.epoch = epoch
        self.loss = dict(
            raw=self.empty_from_dict(losses),
            weighted=self.empty_from_dict(losses),
        )
        self.loss_total = {"raw_total": 0.0, "weighted_total": 0.0}
        # count number of additions to each loss. This is equal to the number
        # of steps per epoch unless some losses are not calculated in all
        # epochs
        self.loss_counter = self.empty_from_dict(losses)

        self.steps = 0
        self.epoch_start_time = perf_counter()

        headers = ("Epoch", "Time (s)", "Raw (total)", "Weighted (total)")
        col_width = (6, 8, 15, 15)
        self._header = "   ".join(f"{j:>{i}s}" for i, j in zip(col_width, headers))

        # keys = ("{epoch}", "{raw_total}", "{weighted_total}")
        # fmt = ("d", ".4f", ".4f")
        self._print_format = (
            f"\033[1m {{{'epoch'}:{col_width[0]}d}}\033[00m   "
            f"{{{'time'}:{col_width[1]}.2f}}   "
            f"\033[95m {{{'raw_total'}:{col_width[2]}.4f}}\033[00m   "
            f"\033[96m {{{'weighted_total'}:{col_width[3]}.4f}}\033[00m   "
        )

        self._header_losses = "   ".join(
            f"{j:>{i}s}" for i, j in zip(col_width, headers)
        )
        self._loss_fmt = "\033[32m {:12s}\033[00m {:10.5f}"

    @staticmethod
    def empty_from_dict(d: dict, value=0.0):
        return {k: {kk: value for kk in v} for k, v in d.items()}

    @staticmethod
    def itemize_dict(d: dict):
        """Call .item() on all entries"""
        return {k: {kk: vv.item() for kk, vv in v.items()} for k, v in d.items()}

    def loss_update(self, loss, wloss):
        self._add_loss(self.loss["raw"], self.itemize_dict(loss))
        self._add_loss(self.loss["weighted"], self.itemize_dict(wloss))
        self._increment_count(self.loss_counter, self.loss["raw"])
        self.steps += 1

    def loss_sum(self):
        self.loss_total["raw_total"] = recursive_dict_sum(self.loss["raw"])
        self.loss_total["weighted_total"] = recursive_dict_sum(self.loss["weighted"])

    def loss_normalize(self):
        self._normalize_dict_by_count(self.loss["raw"], self.loss_counter)
        self._normalize_dict_by_count(self.loss["weighted"], self.loss_counter)

    def print(self):
        if self.epoch == 1:
            print(self._header)
        t_epoch = perf_counter() - self.epoch_start_time
        print(
            self._print_format.format(
                epoch=self.epoch,
                time=t_epoch,
                **self.loss_total,
            )
        )

        # details
        print(
            " | ".join(
                [
                    self._loss_fmt.format(f"{k}[{kk}]", vv)
                    for k, v in self.loss["raw"].items()
                    for kk, vv in v.items()
                ]
            )
        )
        print(
            " | ".join(
                [
                    self._loss_fmt.format(f"{k}[{kk}]", vv)
                    for k, v in self.loss["weighted"].items()
                    for kk, vv in v.items()
                ]
            )
        )

    @staticmethod
    def _add_loss(d0, d1):
        """Update dict of dicts"""
        for k, v in d1.items():
            for kk, vv in v.items():
                d0[k][kk] += vv

        # for k, v in d1.items():
        #     try:
        #         d0[k] += v
        #     except KeyError:
        #         d0[k] = v

        # if k in d0:
        #     d0[k] += v
        # else:
        #     d0[k] = v

    @staticmethod
    def _increment_count(counter, d0):
        for k, v in d0.items():
            for kk in v:
                counter[k][kk] += 1
        # for k in d0:
        #     try:
        #         counter[k] += 1
        #     except KeyError:
        #         counter[k] = 1

        #     if k in counter:
        #         counter[k] += 1
        #     else:
        #         counter[k] = 1

    @staticmethod
    def _normalize_dict_by_count(d, counter):
        for k, v in d.items():
            for kk in v:
                d[k][kk] /= counter[k][kk]


def _flat_item_dict(d: dict, out: None | dict = None, prefix=None, sep=":"):
    """Flattens a dict and calls .item() on its values."""
    if out is None:
        out = {}
    for k, v in d.items():
        key = k if prefix is None else sep.join((prefix, k))
        if isinstance(v, dict):
            _flat_item_dict(v, out, key)
        else:
            out[key] = v.item()
    return out


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="Configuration file defining the training setup."
    )
    parser.add_argument("--resume-wandb", action="store_true")
    # parser.add_argument("--dataset-config", default="dataset.yaml", help="Dataset configuration file.")

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)

    print("Using configuration files")
    print(f"Training    {args.config}")

    config_train = load_config(args.config)
    config_train.wandb.resume |= args.resume_wandb

    # print("wandb config", config_train.wandb)

    trainer = BrainNetTrainer(config_train)
    trainer.train()
