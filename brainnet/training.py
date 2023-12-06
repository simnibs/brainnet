from copy import deepcopy
from pathlib import Path

import torch
import wandb

import brainsynth

from brainnet.modules.brainnet import BrainNet
from brainnet.modules.criterion import Criterion
import brainnet.utilities

config_name = "train_foundation.yaml"
config_ds_name = "datasets.yaml"

_epoch_fmt = lambda epoch: f"{epoch:05d}"

config = brainsynth.config.load_config(brainnet.config_dir / config_name)
config_ds = brainsynth.config.load_config(brainnet.config_dir / config_ds_name)

# Load full images and a random surface

def setup_dataloader_for_synthesizer(config, config_ds):
    ds_dir = Path(config.dataset.dir)

    # Individual datasets
    datasets = [
        brainsynth.dataset.CroppedDataset(
            ds_dir / ds,
            brainsynth.dataset.load_dataset_subjects(ds_dir, ds),
            default_images=config_ds.default_images,
            optional_images=getattr(config_ds.datasets, ds),
            **vars(config.dataset.kwargs),
        ) for ds in config.dataset.train
    ]
    # Concatenated
    dataset = torch.utils.data.ConcatDataset(datasets)
    # Split in train, validation, etc.
    dataset = brainsynth.dataloader.split_dataset(
        dataset, vars(config_ds.dataset_split.splits), config_ds.dataset_split.rng_seed
    )

    kwargs = vars(config.dataloader)
    dataloader = {
        k: brainsynth.dataloader.make_dataloader(v, **kwargs) for k, v in dataset.items()
    }

    return dataloader

# original datasets are in
# dataloader["train"].dataset.dataset.datasets
#                     subset  concat  list of original ds
next(iter(dataloader["train"]))

a = next(iter(dataloader["train"]))

# images, surfaces, info = next(iter(dataset))


self = brainsynth.Synthesizer(device=device)
data = self(images, surfaces, info)


class BrainNetTrainer:
    def __init__(self, config):
        self.config = config

        self.device = torch.device(config.device.model)
        self.synth_device = torch.device(config.device.synth)

        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.epoch.start = self.epoch.resume_from_checkpoint or 1

        self.manual_schedule = config.manual_schedule
        self.loss_scheduler = config["loss_scheduler"]
        self.task_scheduler = config["task_scheduler"]

        self.latest_checkpoint = None
        self.results_dir = Path(config.results.dir)
        self.checkpoint_dir = self.results_dir / config.results/checkpoint

        self.convergence = config["convergence"]

        # Logging
        self.wandb = wandb

        # ...

        self.model = BrainNet(config["feature_extractor"], config["tasks"]).to(
            self.device
        )
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # print("Number of (trainable) parameters in model is", n_parameters)

        self.criterion = Criterion(config["loss"], config["loss_weights"])

        self.optimizer = getattr(torch.optim, config["optimizer"]["name"])(
            self.model.parameters(), config["optimizer"]["kwargs"]
        )
        self.lr_schedulers = torch.optim.lr_scheduler.ChainedScheduler(
            [
                getattr(torch.optim.lr_scheduler, k)(self.optimizer, **v)
                for k, v in config["lr_schedulers"].items()
            ]
        )

    def has_converged(self):
        current_lr = self.optimizer.param_groups[0].lr
        return self.convergence["minimum_lr"] < current_lr

    def get_hyper_params(self):
        return dict(
            LR=self.optimizer.param_groups[0].lr,
        )

    def train(self):
        """Train a model until 'convergence' or maximum number of epochs is
        reached.
        """
        self.model.train()

        for n in range(self.epoch["start"], self.epoch["max_allowed_epochs"] + 1):
            epoch = Epoch(n)

            hyper_params = self.get_hyper_params()

            for step in range(self.epoch["steps_per_epoch"]):
                # choose a random subject from the training pool
                # dataloader stuff here...

                # apply synth: transform etc.

                if (
                    torch.rand([1], device=config.device.synthesizer)
                    < config.synth.probability
                ):
                    data = Synthesizer()
                else:
                    data = images
                    data["synth"] = data["t1"]  # or whatever contrast?

                image = data["image"]

                loss = self.step(image, y_true)

                # reset gradients in optimizer
                self.optimizer.zero_grad()

                # compute weighted loss
                wloss = self.criterion.apply_weights(loss)
                wloss_sum = brainnet.utilities.recursive_dict_sum(wloss)
                # compute gradients
                wloss_sum.backward()

                # update parameters
                self.optimizer.step()

                # log loss
                epoch.loss_update(brainnet.utilities.flatten_dict(loss))

                self.hook_on_step_end(step)

            epoch.loss_normalize_and_sum()

            val_loss = self.validate(epoch.epoch)

            self._wandb_log(epoch.loss, val_loss)  # , hyper_params)

            self.save_checkpoint(n)
            self.hook_on_epoch_end(n)

            if self.has_converged():
                self.save_checkpoint(n)
                print("Converged")
                break

        self._wandb_finish()

    @staticmethod
    def _initialize_loss_dict():
        """Initialize a dictionary to hold aggregated losses for an epoch."""
        return {
            "raw": {},
            "weighted": {},
            "raw (sum)": 0,
            "weighted (sum)": 0,
        }

    def validate(self, epoch):
        if (epoch == self.epoch["start"]) or (
            epoch % self.epoch["validate_every"] != 0
        ):
            return

        val_epoch = Epoch(epoch)

        self.model.eval()
        val_loss = self._initialize_loss_dict()

        with torch.inference_mode():
            for i in ...:
                # choose a random subject from validation pool

                loss = self.step(data, y_true)

                epoch.loss_update(brainnet.utilities.flatten_dict(loss))
            val_epoch.loss_normalize_and_sum()

        for scheduler in self.lr_schedulers:
            scheduler.step(val_loss[...])

        self.hook_on_validation_end(epoch, epoch.loss)

        return epoch.loss

    def load_checkpoint(self, epoch, load_model=True, load_optimizer=True):
        """Load parameters from a state dict."""
        if load_model:
            self.model.load_state_dict(
                torch.load(self.checkpoint_dir / f"{_epoch_fmt(epoch):s}_model.pt")
            )
        if load_optimizer:
            self.optimizer.load_state_dict(
                torch.load(self.checkpoint_dir / f"{_epoch_fmt(epoch):s}_optim.pt")
            )

    def save_checkpoint(self, epoch):
        if (epoch % self.epoch["save_state_every"] != 0) or (
            epoch == self.epoch["start"]
        ):
            return
        torch.save(
            self.model.state_dict(),
            self.checkpoint_dir / f"{_epoch_fmt(epoch):s}_model.pt",
        )
        torch.save(
            self.optimizer.state_dict(),
            self.checkpoint_dir / f"{_epoch_fmt(epoch):s}_optim.pt",
        )
        self.latest_checkpoint = epoch

    def step(self, data, y_true):
        y_pred = self.model(data)
        return self.criterion(y_pred, y_true)

    def hook_on_epoch_end(self, epoch):
        if epoch in self.task_scheduler:
            task_config = deepcopy(self.config["tasks"])
            brainnet.utilities.recursive_dict_update_(
                task_config, self.task_scheduler[epoch]
            )
            m = BrainNet(self.config["feature_extractor"], task_config)
            m.load_state_dict(self.model.state_dict())
            self.model = m

        if epoch in self.loss_scheduler:
            self.criterion.update_weights(self.loss_scheduler[epoch])

    def hook_on_step_end(self, step):
        pass

    def hook_on_validation_end(self, epoch, result):
        if result is not None:
            self.lr_schedulers.step(epoch)

    def _wandb_init(self, config):
        if self.wandb is not None:
            self.run = self.wandb.init(
                project=config["PROJECT_NAME"],
                # entity=,
                config={},
                dir=config["wandb"]["dir"],
                name=config["wandb"]["name"],
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
            data = {"Loss (train)": train_loss}
            if val_loss:
                data["Loss (validation)"] = val_loss
            if hyper_params is not None:
                data["Hyper parameters"] = hyper_params
            self.wandb.log(data, **kwargs)

    def _wandb_finish(self):
        if self.wandb is not None:
            self.wandb.finish()


class Epoch:
    def __init__(self, epoch) -> None:
        self.epoch = epoch
        self.loss = {"raw": {}, "weighted": {}, "raw (sum)": 0.0, "weighted (sum)": 0.0}
        # count number of additions to each loss. This is equal to the number
        # of steps per epoch unless some losses are not calculated in all
        # epochs
        self.loss_counter = {}

    def loss_update(self, loss):
        self._update_dict(self.loss["raw"], loss)
        self._update_dict(self.loss["weighted"], loss)
        self._update_key_count(self.loss_counter, loss)

    def loss_sum(self):
        self.loss["raw (sum)"] = sum(self.loss["raw"].values())
        self.loss["weighted (sum)"] = sum(self.loss["weighted"].values())

    def loss_normalize_and_sum(self):
        self._normalize_dict_by_count(self.loss["raw"], self.loss_counter)
        self._normalize_dict_by_count(self.loss["weighted"], self.loss_counter)
        self.loss_sum()

    @staticmethod
    def _update_dict(d0, d1):
        for k, v in d1.items():
            if k in d0:
                d0[k] += v
            else:
                d0[k] = v  # .item()

    @staticmethod
    def _update_key_count(counter, d0):
        for k, v in d0.items():
            if k in counter:
                counter[k] += v
            else:
                counter[k] = v

    @staticmethod
    def _normalize_dict_by_count(d, count):
        for k in d:
            d[k] /= count[k]
