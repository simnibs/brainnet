import torch

import brainnet.utilities
from brainnet.modules import loss_wrappers


class Criterion(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # self.weight_threshold = weight_threshold

        self.weights = vars(config.weights)

        self.losses = {
            name: self.setup_loss(loss_config)
            for name, loss_config in vars(config.functions).items()
        }

    @staticmethod
    def setup_loss(config):
        # assert "module" in kwargs, "Loss definition should contain `module` definition"
        # assert "loss" in kwargs, "Loss definition should contain `loss` definition"

        module = config.module.name
        module_kw = vars(config.module.kwargs)

        loss_fn = config.loss.name
        loss_kw = vars(config.loss.kwargs) if hasattr(config.loss, "kwargs") else None

        return getattr(loss_wrappers, module)(
            loss_fn, **module_kw, loss_fn_kwargs=loss_kw
        )


    # def blabla(self):

    #     if

    def forward(self, y_pred, y_true, **kwargs):
        """Compute all losses that is possible given the entries in `y_pred`"""
        weight_total = 0.0

        # Compute raw loss
        loss_dict = {}
        for name, loss in self.losses.items():
            try:
                match loss:
                    case loss_wrappers.SupervisedLoss:
                        # if isinstance(loss, brainnet.modules.losses.SymmetricMSELoss):
                        #     i_pred = loss.loss_fn.i_pred
                        #     i_true = loss.loss_fn.i_true
                        #     curv_true = kwargs["curv_true"]
                        value = loss(y_pred, y_true)
                    case loss_wrappers.RegularizationLoss:
                        value = loss(y_pred)
                    case _:
                        raise ValueError
                loss_dict[name] = value
                weight_total += self.weights[name]
            except KeyError:
                # Required data does not exist in y_pred and/or y_true
                pass

        # Compute weighted loss
        weight_normalizer = 1 / weight_total
        weighted_loss_dict = {
            k: v * self.weights[k] * weight_normalizer for k, v in loss_dict.items()
        }

        return loss_dict, weighted_loss_dict


class oldCriterion(torch.nn.Module):
    def __init__(self, config, datasets, weight_threshold=1e-8) -> None:
        super().__init__()

        self.weight_threshold = weight_threshold

        self.weights = recursive_namespace_to_dict(config.weights)
        if config.normalize_weights:
            self._normalize_weights()

        self.losses = {
            task: {name: self.get_loss(task, loss) for name, loss in losses.items()}
            for task, losses in recursive_namespace_to_dict(config.functions).items()
        }

        # compute loss normalizer per dataset
        weights_sum_tasks = {
            task: brainnet.utilities.recursive_dict_sum(losses)
            for task, losses in self.weights.items()
        }

        # weights = brainnet.utilities.flatten_dict(weights_config)
        # weight_names = tuple(weights.keys())

        # weight_tensor = torch.tensor(tuple(weights.values()), device=device)
        # self.n_losses = len(weight_tensor)

        # loss_index[ds] = {
        #     ds: torch.tensor(
        #         [self._valid_task_for_dataset(name[0], avail) for name in weight_names],
        #         dtype=torch.bool,
        #         device=device
        #     ).nonzero().ravel() for ds,avail in datasets.items()
        # }

        # determine which tasks are relevant for which datasets
        self.tasks_per_ds = {
            ds: [
                task
                for task in self.weights
                if self._valid_task_for_dataset(task, avail)
            ]
            for ds, avail in recursive_namespace_to_dict(datasets).items()
        }

        self.dataset_normalizer = {}
        for ds, tasks in self.tasks_per_ds.items():
            self.dataset_normalizer[ds] = 0.0
            for task in tasks:
                self.dataset_normalizer[ds] += weights_sum_tasks[task]

    @staticmethod
    def _valid_task_for_dataset(task, available_images):
        return (task not in optional_images) or (task in available_images)

    @staticmethod
    def get_loss(task, kwargs):
        assert "module" in kwargs, "Loss definition should contain `module` definition"
        assert "loss" in kwargs, "Loss definition should contain `loss` definition"
        module = kwargs["module"]
        kwargs_ = {k: v for k, v in kwargs.items() if k != "module"}
        if "y_pred" not in kwargs_:
            kwargs_["y_pred"] = task
        if "y_true" not in kwargs:
            kwargs_["y_true"] = task
        return getattr(brainnet.modules.loss_wrappers, module)(**kwargs_)

    # def update_weights(self, weights):
    #     # we need the copy because self.weights are normalized and weights are not
    #     self.weights = deepcopy(self._original_weights)
    #     brainnet.utilities.recursive_dict_update_(self.weights, weights)
    #     self._normalize_weights()

    def _normalize_weights(self):
        total = brainnet.utilities.recursive_dict_sum(self.weights)
        brainnet.utilities.recursive_dict_multiply(self.weights, 1 / total)

    @staticmethod
    def flatten_loss(loss):
        return brainnet.utilities.flatten_dict(loss)

    def apply_weights(self, loss):
        """"""
        return brainnet.utilities.multiply_dicts(loss, self.weights)

    def forward(self, y_pred, y_true, ds_id):
        """Compute all losses."""
        out_loss = {}
        out_loss_weighted = {}

        for task in self.tasks_per_ds[ds_id]:
            x, y = self._compute_task_loss(y_pred, y_true, task, ds_id)
            out_loss[task] = x
            out_loss_weighted[task] = y

        return out_loss, out_loss_weighted

    def _compute_task_loss(self, y_pred, y_true, task, ds_id):
        """Compute all losses for a specific task."""
        normalizer = 1.0 / self.dataset_normalizer[ds_id]
        task_loss = {}
        task_loss_weighted = {}
        for name, loss in self.losses[task].items():
            match loss:
                case brainnet.modules.loss_wrappers.SurfaceLoss:
                    # this is a little special as SurfaceLoss returns a dict

                    # name: here is white or pial
                    # e.g., compute loss for white across "all" predicted
                    # hemispheres
                    task_loss[name] = 0.0
                    for hemi in y_pred[task]:
                        task_loss[name] += loss(y_pred[task][hemi], y_true[task][hemi])
                    task_loss[name] /= len(y_pred[task])

                case _:
                    if (w := self.weights[task][name]) > self.weight_threshold:
                        value = loss(y_pred, y_true)
                        task_loss[name] = value
                        task_loss_weighted[name] = value * w * normalizer
                    else:
                        task_loss[name] = 0.0
                        task_loss_weighted[name] = 0.0

        #     if isinstance(loss, dict):
        #         task_loss[name] = {}
        #         task_loss_weighted[name] = {}
        #         for k, v in loss.items():
        #             if (w := self.weights[task][name][k]) > self.weight_threshold:
        #                 value = v(y_pred, y_true)
        #                 task_loss[name][k] = value
        #                 task_loss_weighted[name][k] = value * w * normalizer
        #             else:
        #                 task_loss[name][k] = 0.0
        #                 task_loss_weighted[name][k] = 0.0
        #     else:
        #         if (w := self.weights[task][name]) > self.weight_threshold:
        #             value = loss(y_pred, y_true)
        #             task_loss[name] = value
        #             task_loss_weighted[name] = value * w * normalizer
        #         else:
        #             task_loss[name] = 0.0
        #             task_loss_weighted[name] = 0.0

        return task_loss, task_loss_weighted

    # @staticmethod
    # def _forward(y_pred, y_true, losses, weights, out):
    #     if out is None:
    #         out = {}
    #     for k,v in losses.items():
    #         if isinstance(v, dict):
    #             out[k] = {}
    #             self._forward(y_pred, y_true, out[k], )
    #         else:
    #             out[k] = loss(y_pred, y_true) if self.weights[task][name] > self.weight_threshold else 0
    #     return out
