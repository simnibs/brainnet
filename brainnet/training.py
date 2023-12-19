import argparse
from copy import deepcopy
from datetime import datetime
import sys
from time import perf_counter

import torch
import wandb

import brainsynth

from brainnet.modules.brainnet import BrainNet
from brainnet.modules.tasks import SurfaceModule
from brainnet.modules.criterion import Criterion
import brainnet.utilities

from brainsynth.config.utilities import load_config, recursive_namespace_to_dict
from brainnet.utilities import recursively_apply_function
from brainsynth.dataset import get_dataloader_concatenated_and_split


from brainnet.mesh.surface import BatchedSurfaces
from brainnet.mesh.topology import get_recursively_subdivided_topology

fmt_epoch = lambda epoch: f"{epoch:05d}"
fmt_model = lambda epoch: f"{fmt_epoch(epoch)}_model.pt"
fmt_optim = lambda epoch: f"{fmt_epoch(epoch)}_optim.pt"


# Load full images and a random surface


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

# filename = "/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_segment_t1.yaml"
filename = "/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_foundation.yaml"
config = load_config(filename)
self = BrainNetTrainer(config)

(ds_id, images, surfaces, info) = next(iter(self.dataloader["train"]))

self.to_device(image)
self.to_device(surface)
self.to_device(info)

data = self.synthesizer(image, surface, info)

trainer.train()


# available_datasets = brainsynth.config.load_config("/mrhome/jesperdn/repositories/brainnet/brainnet/config/datasets.yaml")
# available_datasets = available_datasets.datasets
# available_datasets = {ds: getattr(available_datasets, ds) for ds in config.datasets.train}

dataloader = get_dataloader_concatenated_and_split(
            **_config_to_args_dataloader(config)
        )

device = torch.device(config.device.model)
image = torch.rand((1,1,2**7,2**7,2**7), device=device)
features = torch.rand((1, 64, 2**6,2**7,2**8), device=device)

model = BrainNet(config.model.feature_extractor, config.model.tasks, device)
model = model.to(device)

model.tasks.surface(features)

y_pred = model(image)

model.tasks(features)

strides = torch.IntTensor([2,2,2])

shape = torch.IntTensor([220,200,180])

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


# next(iter(dataloader["train"]))

# a = next(iter(dataloader["train"]))

# # images, surfaces, info = next(iter(dataset))

# for i in dataloader["train"]:
#     print("dataset", i[0])
#     print("index", i[1])
#     print(i[2])
#     print(i[3])
#     print(i[4])


# self = brainsynth.Synthesizer(device=device)
# data = self(images, surfaces, info)


class BrainNetTrainer:
    def __init__(self, config):
        torch.backends.cudnn.benchmark = True  # increases initial time
        torch.backends.cudnn.deterministic = True

        self.config = config

        self.device = torch.device(config.device.model)
        self.synth_device = torch.device(config.device.synthesizer)

        # assert self.device == self.synth_device

        self.epoch = config.epoch
        self.epoch.start = self.epoch.resume_from_checkpoint or 1

        self.latest_checkpoint = None

        self.results_dir = self.config.results.dir / self.config.PROJECT_NAME
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

        self.checkpoint_dir = self.results_dir / "checkpoints"
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

        assert self.config.dataset.dataloader.batch_size == 1

        self.dataloader = get_dataloader_concatenated_and_split(
            **_config_to_args_dataloader(config)
        )

        print("Dataloaders")
        print(f"# samples in train      {len(self.dataloader['train']):d}")
        print(f"# samples in validation {len(self.dataloader['validation']):d}")

        # We need to ensure that all downsampling steps are valid

        # if self.config.model.feature_extractor.model == "UNet" # or perhaps other..?
        if (
            (
                strides := torch.tensor(
                    self.config.model.feature_extractor.kwargs.strides,
                    dtype=torch.int,
                    device=self.synth_device,
                )
            )
            > 1
        ).any():
            divisor = strides ** len(strides)
        else:
            divisor = None

        self.synthesizer = brainsynth.Synthesizer(
            self.config.synthesizer.config,
            ensure_divisible_by=divisor,
            device=self.synth_device,
        )
        self.synthesizer.to(self.synth_device)

        # Logging
        if hasattr(self.config, "wandb") and self.config.wandb.enable:
            self.wandb = wandb
            self.wandb_dir = self.results_dir  # / "wandb"
            if not self.wandb_dir.exists():
                self.wandb_dir.mkdir(parents=True)
        else:
            self.wandb = None

        # Device is needed as arg for topofit for now...
        self.model = BrainNet(
            config.model.feature_extractor, config.model.tasks, self.device
        )
        self.model.to(self.device)
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters: {n_parameters}")

        self.optimizer = getattr(torch.optim, config.optimizer.name)(
            self.model.parameters(),
            **recursive_namespace_to_dict(config.optimizer.kwargs),
        )

        self.criterion = Criterion(config.loss)

        if self.config.auto_schedulers is not None:
            raise NotImplementedError

            self.lr_schedulers = torch.optim.lr_scheduler.ChainedScheduler(
                [
                    getattr(torch.optim.lr_scheduler, k)(self.optimizer, **v)
                    for k, v in config["lr_schedulers"].items()
                ]
            )

        if self.config.manual_schedulers is not None:
            raise NotImplementedError

        # load model and optimizer parameters
        self.load_checkpoint(self.epoch.resume_from_checkpoint)

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def has_converged(self):
        return self._get_lr() <= self.config.convergence.minimum_lr

    def get_hyper_params(self):
        return dict(
            LR=self.optimizer.param_groups[0].lr,
        )

    def to_device(self, dict_of_tensors):
        for k, v in dict_of_tensors.items():
            if isinstance(v, dict):
                self.to_device(v)
            else:
                dict_of_tensors[k] = v.to(self.device)

    def train(self):
        """Train a model until 'convergence' or maximum number of epochs is
        reached.
        """

        now = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        print(f"Started training on {now}")

        self._wandb_init()

        # Get empty BatchedSurfaces. We update the vertices at each iteration
        self.surface_skeletons = dict(
            y_pred=self.get_surface_skeletons(),
            y_true=self.get_surface_skeletons(),
        )

        for n in range(self.epoch.start, self.epoch.max_allowed_epochs + 1):
            epoch = Epoch(n)

            # hyper_params = self.get_hyper_params()
            print("epoch", n)

            self.model.train()
            for step, (ds_id, images, surfaces, info) in enumerate(
                self.dataloader["train"]
            ):
                if step == self.epoch.steps_per_epoch:
                    break

                t0 = perf_counter()

                print("step", step)

                # synth and model device could be different!

                ds_id = ds_id[0]  # ds_id is a tuple of len 1
                # self.to_device(image)
                # self.to_device(surface)
                # self.to_device(info)

                with torch.no_grad():
                    y_true_img, y_true_surf = self.synthesizer(images, surfaces, info)

                    if "synth" not in y_true_img:
                        # select a random contrast from the list of alternative images
                        avail = getattr(self.config.dataset.alternative_synth, ds_id)
                        sel = torch.randint(
                            0, len(avail), (1,), device=self.synthesizer.device
                        )
                        y_true_img["synth"] = y_true_img[avail[sel]]

                    # do it AFTER synth for now...
                    self.to_device(y_true_img)
                    self.to_device(y_true_surf)

                    image = y_true_img.pop("synth")
                    y_true = y_true_img
                    y_true["surface"] = y_true_surf

                # convert surface ground truths to batched surfaces
                # ...

                # also load precomputed curvature for true...

                t1 = perf_counter()

                if len(surfaces) > 0:
                    model_kwargs = dict(hemispheres=tuple(y_true_surf.keys()))
                else:
                    model_kwargs = {}

                # add y_true_curv here
                # criterion_kwargs = dict(y_true_curv=surfaces[...])

                # with torch.autograd.set_detect_anomaly(True):
                loss, wloss = self.step(image, y_true, model_kwargs)  # criterion_kwargs

                t2 = perf_counter()

                # compute weighted loss
                # wloss = self.criterion.apply_weights(loss)
                wloss_sum = sum(wloss.values())

                # Reset gradients in optimizer. Otherwise gradients would
                # accumulate across multiple passes (whenever .backward is
                # called)
                self.optimizer.zero_grad()
                # Compute and accumulate gradients. backward() frees
                # intermediate values of the graph (e.g., activations)

                # torch.cuda.empty_cache()
                # print_memory_usage(self.device)

                wloss_sum.backward()
                # Update parameters (i.e., gradients)
                self.optimizer.step()

                torch.cuda.empty_cache()
                # print_memory_usage(self.device)


                # log the loss
                epoch.loss_update(loss, wloss)

                self.hook_on_step_end(step)

                t3 = perf_counter()

                synth_time = t1 - t0
                model_time = t2 - t1
                step_time = t3 - t0

                print("synth time", synth_time)
                print("model time", model_time)
                print("step  time", step_time)

            epoch.loss_normalize()
            epoch.loss_sum()

            epoch.print()

            val_loss = self.validate(epoch.epoch)

            self._wandb_log(epoch.loss, val_loss)  # , hyper_params)

            self.save_checkpoint(n)
            self.hook_on_epoch_end(n)

            if self.has_converged():
                self.save_checkpoint(n)
                print("Converged")
                break

        self._wandb_finish()

    def step(self, image, y_true, model_kwargs=None, criterion_kwargs=None):
        model_kwargs = model_kwargs or {}
        criterion_kwargs = criterion_kwargs or {}

        y_pred = self.model(image, **model_kwargs)

        # convert surface predictions to batched surfaces
        if (k := "surface") in y_pred:
            self.set_batchedsurface(y_pred[k], self.surface_skeletons["y_pred"])
            self.set_batchedsurface(y_true[k], self.surface_skeletons["y_true"])

            self.criterion.precompute_for_surface_loss(y_pred[k], y_true[k])

        loss = self.criterion(y_pred, y_true, **criterion_kwargs)
        wloss = self.criterion.apply_normalized_weights(loss)

        return loss, wloss

    def _do_save_checkpoint(self, epoch):
        return epoch % self.epoch.save_state_every == 0

    def _do_validation(self, epoch):
        return epoch % self.epoch.validate_every == 0

    def validate(self, epoch):
        if not self._do_validation(epoch):
            return {}

        val_epoch = Epoch(epoch)

        self.model.eval()
        with torch.inference_mode():
            for step, (ds_id, images, surfaces, info) in enumerate(
                self.dataloader["validation"]
            ):
                if step == self.epoch.steps_per_validation:
                    break

                ds_id = ds_id[0]
                self.to_device(images)
                self.to_device(surfaces)
                self.to_device(info)

                # FIXME what image is used for evaluation???
                image = images.pop("norm")
                y_true = images
                y_true["surface"] = surfaces

                # same as training step for now...
                loss, wloss = self.criterion(self.model(image, y_true), y_true, ds_id)

                val_epoch.loss_update(loss, wloss)

            val_epoch.loss_normalize()
            val_epoch.loss_sum()

        self.hook_on_validation_end(val_epoch, val_epoch.loss)

        val_epoch.print()

        return val_epoch.loss

    def load_checkpoint(self, epoch, load_model=True, load_optimizer=True):
        """Load parameters from a state dict."""
        if epoch is None:
            return

        if load_model:
            self.model.load_state_dict(
                torch.load(self.checkpoint_dir / fmt_model(epoch))
            )
        if load_optimizer:
            self.optimizer.load_state_dict(
                torch.load(self.checkpoint_dir / fmt_optim(epoch))
            )

    def save_checkpoint(self, epoch):
        if not self._do_save_checkpoint(epoch):
            return

        torch.save(
            self.model.state_dict(),
            self.checkpoint_dir / fmt_model(epoch),
        )
        torch.save(
            self.optimizer.state_dict(),
            self.checkpoint_dir / fmt_optim(epoch),
        )
        self.latest_checkpoint = epoch

    def get_surface_skeletons(self):
        surface_names = ("white", "pial")

        module = [i for i in self.model.tasks.values() if isinstance(i, SurfaceModule)]
        if len(module) == 0:
            return None

        assert len(module) == 1

        module = module[0]

        topology = module.get_prediction_topology()
        topology = dict(lh=topology, rh=deepcopy(topology))
        topology["rh"].reverse_face_orientation()

        return {
            h: {
                s: BatchedSurfaces(torch.zeros(t.n_vertices, 3, device=self.device), t)
                for s in surface_names
            }
            for h, t in topology.items()
        }

    def set_batchedsurface(self, data, surface_skeleton):
        """Replace vertex tensors with BatchedSurfaces objects."""
        for h, surfaces in data.items():
            for s in surfaces:
                # convert metatensor to tensor otherwise the custom CUDA
                # functions fail (nearest neighbor computation)
                surface_skeleton[h][s].vertices = surfaces[s]
                data[h][s] = surface_skeleton[h][s]

    def hook_on_step_end(self, step):
        pass

    def hook_on_epoch_end(self, epoch):
        pass

        # epoch_ = str(epoch)

        # # Re-initialize BrainNet using the updated task parameters
        # if epoch_ in self.config.manual_schedulers.tasks:
        #     task_config = deepcopy(self.config["tasks"])
        #     brainnet.utilities.recursive_dict_update_(
        #         task_config, getattr(self.config.manual_schedulers.tasks, epoch_)
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

        # if result is not None:
        #     self.lr_schedulers.step(epoch)

    def _wandb_init(self):
        if self.wandb is not None:
            self.run = self.wandb.init(
                project=self.config.wandb.project_name,
                # entity=,
                config={},
                dir=self.wandb_dir,
                name=self.config.wandb.name,
            )

    def _wandb_log(
        self,
        train_loss: dict,
        val_loss: dict | None = None,
        hyper_params: dict | None = None,
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
            self.wandb.log(data, **kwargs)

    def _wandb_finish(self):
        if self.wandb is not None:
            self.wandb.finish()


class Epoch:
    def __init__(self, epoch) -> None:
        self.epoch = epoch
        self.loss = {"raw": {}, "weighted": {}}
        self.loss_total = {"raw_total": 0.0, "weighted_total": 0.0}
        # count number of additions to each loss. This is equal to the number
        # of steps per epoch unless some losses are not calculated in all
        # epochs
        self.loss_counter = {}

        headers = ("Epoch", "Raw (total)", "Weighted (total)")
        col_width = (6, 15, 15)
        self._header = "   ".join(f"{j:>{i}s}" for i, j in zip(col_width, headers))

        # keys = ("{epoch}", "{raw_total}", "{weighted_total}")
        # fmt = ("d", ".4f", ".4f")
        self._print_format = (
            f"{{{'epoch'}:{col_width[0]}d}}   "
            f"{{{'raw_total'}:{col_width[1]}.4f}}   "
            f"{{{'weighted_total'}:{col_width[2]}.4f}}   "
        )

    def loss_update(self, loss, wloss):
        self._update_dict(self.loss["raw"], {k: v.item() for k, v in loss.items()})
        self._update_dict(
            self.loss["weighted"],
            {k: v.item() for k, v in wloss.items()},
        )
        self._update_key_count(self.loss_counter, self.loss["raw"])

    def loss_sum(self):
        self.loss_total["raw_total"] = sum(self.loss["raw"].values())
        self.loss_total["weighted_total"] = sum(self.loss["weighted"].values())

    def loss_normalize(self):
        self._normalize_dict_by_count(self.loss["raw"], self.loss_counter)
        self._normalize_dict_by_count(self.loss["weighted"], self.loss_counter)

    def print(self):
        if self.epoch == 1:
            print(self._header)
        print(self._print_format.format(epoch=self.epoch, **self.loss_total))

    @staticmethod
    def _update_dict(d0, d1):
        for k, v in d1.items():
            if k in d0:
                d0[k] += v
            else:
                d0[k] = v

    @staticmethod
    def _update_key_count(counter, d0):
        for k in d0:
            if k in counter:
                counter[k] += 1
            else:
                counter[k] = 1

    @staticmethod
    def _normalize_dict_by_count(d, count):
        for k in d:
            d[k] /= count[k]


def _flat_item_dict(d: dict, out: None | dict = None, prefix=None, sep=":"):
    """Flattens a dict and calls .item() on its values."""
    if out is None:
        out = {}
    for k, v in d.items():
        key = k if prefix is None else sep.join((prefix, k))

        # match prefix:
        #     case None:
        #         key = k
        #     case tuple():
        #         key = (*prefix, k)
        #     case _:
        #         key = (prefix, k)

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
    # parser.add_argument("--dataset-config", default="dataset.yaml", help="Dataset configuration file.")

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)

    print("Using configuration files")
    print(f"Training    {args.config}")

    config_train = load_config(args.config)

    trainer = BrainNetTrainer(config_train)
    trainer.train()
