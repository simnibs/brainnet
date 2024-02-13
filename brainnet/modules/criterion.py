import torch

from brainnet.modules import loss_wrappers
from brainnet.modules.losses import (
    MatchedCurvatureLoss,
    SymmetricMeanSquaredNormLoss,
    SymmetricCurvatureLoss,
)
from brainsynth.config.utilities import recursive_namespace_to_dict


class Criterion(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # self.weight_threshold = weight_threshold

        self.task_weights = recursive_namespace_to_dict(config.task_weights)
        self.loss_weights = recursive_namespace_to_dict(config.loss_weights)

        # across_task_normalizer is computed on every forward pass depending on
        # which task losses are feasible
        self.within_task_normalizer = {
            task: 1 / sum(w for w in losses.values())
            for task, losses in self.loss_weights.items()
        }
        # self.across_task_normalizer = {}

        self.losses = {
            task: {
                loss_name: self.setup_loss(loss_config)
                for loss_name, loss_config in vars(task_losses).items()
            }
            for task, task_losses in vars(config.functions).items()
        }

        self.has_SymmetricLoss = any(
            isinstance(
                loss.loss_fn, (SymmetricMeanSquaredNormLoss, SymmetricCurvatureLoss)
            )
            for task_losses in self.losses.values()
            for loss in task_losses.values()
        )
        self.has_CurvatureLoss = any(
            isinstance(loss.loss_fn, (SymmetricCurvatureLoss, MatchedCurvatureLoss))
            for task_losses in self.losses.values()
            for loss in task_losses.values()
        )

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

    def apply_weights(self, loss_dict):
        """Apply normalized weights. `forward` needs to be run in order to
        update the weight normalizer.

        Each individual loss is weighted as

            loss * loss_weight * within_task_normalizer * across_task_normalizer

        such the loss weights within and across tasks sum to one.
        """
        return {
            task: {
                name: loss
                * self.loss_weights[task][name]
                * self.within_task_normalizer[task]
                * self.task_weights[task]
                # * self.across_task_normalizer[task]
                for name, loss in losses.items()
            }
            for task, losses in loss_dict.items()
        }

    def precompute_for_surface_loss(self, y_pred: dict, y_true: dict):
        """Precompute useful things for calculating the losses."""

        index_name = "index"
        curvature_name = "curv"

        n_smooth = 1

        if self.has_SymmetricLoss:  # chamfer *or* curvature
            # compute nearest neighbors
            for h, surfaces in y_pred.items():
                for s in surfaces:
                    # these are indices into y_true!
                    index = y_pred[h][s].nearest_neighbor(y_true[h][s])
                    y_pred[h][s].vertex_data[index_name] = index
                    # these are indices into y_pred!
                    index = y_true[h][s].nearest_neighbor(y_pred[h][s])
                    y_true[h][s].vertex_data[index_name] = index

        # compute curvature
        if self.has_CurvatureLoss:
            for h, surfaces in y_pred.items():
                for s in surfaces:
                    curv = y_pred[h][s].compute_mean_curvature_vector()
                    y_pred[h][s].vertex_data[curvature_name] = curv

                    curv = y_true[h][s].compute_mean_curvature_vector()
                    if n_smooth > 0:
                        # smooth the true surface curvature
                        curv = y_true[h][s].compute_iterative_spatial_smoothing(
                            curv, n_smooth
                        )
                    y_true[h][s].vertex_data[curvature_name] = curv

    # def forward(self, y_pred, y_true):
    #     """Compute all losses that is possible given the entries in `y_pred`"""
    #     weight_total = 0.0

    #     # weight_total = {k: 0 for k in self._weight_groups}
    #     # weight_total[self._name_to_weight_group[name]] += self.weights[name]

    #     # Compute raw loss
    #     loss_dict = {}
    #     for name, loss in self.losses.items():
    #         try:
    #             match loss:
    #                 case loss_wrappers.SupervisedLoss():
    #                     # if isinstance(loss, brainnet.modules.losses.SymmetricMSELoss):
    #                     #     i_pred = loss.loss_fn.i_pred
    #                     #     i_true = loss.loss_fn.i_true
    #                     #     curv_true = kwargs["curv_true"]
    #                     value = loss(y_pred, y_true)
    #                 case loss_wrappers.RegularizationLoss():
    #                     value = loss(y_pred)
    #                 case _:
    #                     raise ValueError
    #             loss_dict[name] = value
    #             weight_total += self.weights[name]
    #         except KeyError:
    #             # Required data does not exist in y_pred and/or y_true
    #             pass

    #     # Compute weighted loss
    #     self._weight_normalizer = 1 / weight_total

    #     return loss_dict

    def forward(self, y_pred, y_true):
        """Compute all losses that is possible given the entries in `y_pred`"""

        # Compute raw loss
        loss_dict = {}
        for task, task_losses in self.losses.items():
            loss_dict[task] = {}
            found = False

            for name, loss in task_losses.items():
                # we try as this will usually be okay
                try:
                    match loss:
                        case loss_wrappers.SupervisedLoss():
                            value = loss(y_pred, y_true)
                        case loss_wrappers.RegularizationLoss():
                            value = loss(y_pred)
                        case _:
                            raise ValueError
                    loss_dict[task][name] = value
                    found = True
                except KeyError:
                    # Required data does not exist in y_pred and/or y_true
                    pass

            if not found:
                del loss_dict[task]

        # set the across task normalizer

        # wn = 1 / sum(self.task_weights[t] for t in loss_dict)
        # self.across_task_normalizer = {
        #     t: self.task_weights[t] * wn for t in loss_dict
        # }

        return loss_dict
