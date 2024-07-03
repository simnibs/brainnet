import torch

from brainnet.config import LossParameters
import brainnet.modules

def recursive_dict_setter(d, k, v):
    if len(k) == 1:
        d[k[0]] = v
    else:
        return recursive_dict_setter(d[k[0]], k[1:], v)

class Criterion(torch.nn.Module):

    def __init__(self, config: LossParameters) -> None:
        super().__init__()

        self.loss_functions = config.functions

        self._head_weights = config.head_weights
        self._loss_weights = config.loss_weights
        self._set_active_heads() # sets everything

        # across_task_normalizer is computed on every forward pass depending on
        # which task losses are feasible
        # self.within_task_normalizer = {
        #     task: 1 / sum(w for w in losses.values())
        #     for task, losses in self.loss_weights.items()
        # }
        # self.across_task_normalizer = {}



        # self.lambda_within = {
        #     task: {
        #         loss_name: self.setup_loss(loss_config)
        #         for loss_name, loss_config in vars(task_losses).items()
        #     }
        #     for task, task_losses in vars(config.functions).items()
        # }

        # within_task_normalizer = {
        #     task: 1 / sum(w for w in losses.values())
        #     for task, losses in self.loss_weights.items()
        # }
        # self.intra_task_lambda = {
        #     task: torch.nn.ParameterDict({k: v * within_task_normalizer[task] for k,v in losses})
        #     for task, losses in self.loss_weights.items()
        # }

    def _set_active_heads(self):
        self._active_heads = [h for h,v in self._head_weights.items() if v > 0.0]
        # if any heads changed status we need to update active losses
        self._set_active_losses()

    def _set_active_losses(self):
        # if a head is inactive, all its losses are ignored
        self._active_losses = {
            head: [n for n,v in self._loss_weights[head].items() if v > 0.0]
            for head in self._active_heads
        }
        self._set_needs_sampling()

    def _set_needs_sampling(self):
        """We keep track of this because we can avoid some calculations (e.g.,
        sampling points and finding nearest neighbors) when there is no active
        chamfer/curvature loss.
        """
        self._needs_sampling = any(
            isinstance(
                self.loss_functions[head][loss].loss_fn,
                (
                    brainnet.modules.losses.AsymmetricChamferLoss,
                    brainnet.modules.losses.AsymmetricCurvatureNormLoss,
                    brainnet.modules.losses.SymmetricChamferLoss,
                    brainnet.modules.losses.SymmetricCurvatureNormLoss,
                ),
            )
            for head, v in self._active_losses.items() for loss in v
        )

    def update_head_weights(self, weights):
        for k,v in weights.items():
            if isinstance(k, (list,tuple)):
                recursive_dict_setter(self._head_weights, k, v)
            else:
                self._head_weights[k] = v
        self._set_active_heads()

    def update_loss_weights(self, weights):
        for k,v in weights.items():
            if isinstance(k, (list,tuple)):
                recursive_dict_setter(self._loss_weights, k, v)
            else:
                self._loss_weights[k] = v
        self._set_active_losses()

    # @staticmethod
    # def setup_loss(config):
    #     # assert "module" in kwargs, "Loss definition should contain `module` definition"
    #     # assert "loss" in kwargs, "Loss definition should contain `loss` definition"

    #     module = config.module.name
    #     module_kw = vars(config.module.kwargs)

    #     loss_fn = config.loss.name
    #     loss_kw = vars(config.loss.kwargs) if hasattr(config.loss, "kwargs") else None

    #     return getattr(loss_wrappers, module)(
    #         loss_fn, **module_kw, loss_fn_kwargs=loss_kw
    #     )

    def apply_weights(self, loss_dict):
        """Apply normalized weights. `forward` needs to be run in order to
        update the weight normalizer.

        Each individual loss is weighted as

            loss * loss_weight * within_task_normalizer * across_task_normalizer

        such the loss weights within and across tasks sum to one.
        """
        return {
            head: {
                loss: value * self._loss_weights[head][loss]
                # * self.within_task_normalizer[task]
                * self._head_weights[head]
                # * self.across_task_normalizer[task]
                for loss, value in losses.items()
            }
            for head, losses in loss_dict.items()
        }

    def prepare_for_surface_loss(
        self,
        y_pred: dict,
        y_true: dict,
        n_samples=100000,
        smooth_y_true=True,
    ):
        """Precompute useful things for calculating the losses."""
        # smooth_y_true = False #True # apply smoothing to y_true before calculating K (and H)

        if not self._needs_sampling:
            return

        # n_samples = self.config.prepare_for_surface_loss.n_samples
        # smooth_y_true = self.config.prepare_for_surface_loss.smooth_y_true
        # curv_weight = self.config.prepare_for_surface_loss.curv_weight

        # clip H of y_true before interpolating to sampled points
        H_clip_to_percentile = dict(
            white = (0.001, 0.999),
            pial = (0.01, 0.99),
        )

        index_name = "index_sampled"
        sample_name = "points_sampled"
        K_name = "K_sampled"
        H_name = "H_sampled"

        W_name = "weights_sampled"

        for h, surfaces in y_pred.items():
            for s in surfaces:

                sampled_points, sampled_K, sampled_H, _ = self._sample_points_and_curv(
                    y_pred[h][s],
                    n_samples,
                )
                y_pred[h][s].vertex_data[sample_name] = sampled_points
                y_pred[h][s].vertex_data[K_name] = sampled_K
                y_pred[h][s].vertex_data[H_name] = sampled_H

                sampled_points, sampled_K, sampled_H, sampled_weights  = self._sample_points_and_curv(
                    y_true[h][s],
                    n_samples,
                    smooth_y_true,
                    H_clip_to_percentile=H_clip_to_percentile[s]
                )

                y_true[h][s].vertex_data[sample_name] = sampled_points
                y_true[h][s].vertex_data[K_name] = sampled_K
                y_true[h][s].vertex_data[H_name] = sampled_H

                # these are indices into y_true!
                index = y_pred[h][s].nearest_neighbor_tensors(
                    y_pred[h][s].vertex_data[sample_name],
                    y_true[h][s].vertex_data[sample_name],
                )
                y_pred[h][s].vertex_data[index_name] = index

                # these are indices into y_pred!
                index = y_true[h][s].nearest_neighbor_tensors(
                    y_true[h][s].vertex_data[sample_name],
                    y_pred[h][s].vertex_data[sample_name],
                )
                y_true[h][s].vertex_data[index_name] = index

                # NOTE
                # add sampled weights
                if sampled_weights is not None:
                    y_true[h][s].vertex_data[W_name] = sampled_weights
                    y_pred[h][s].vertex_data[W_name] = sampled_weights[y_pred[h][s].batch_ix, y_pred[h][s].vertex_data[index_name]]

    def _sample_points_and_curv(
        self,
        surface,
        n_samples: int,
        taubin_smoothing: bool = False,
        H_clip_to_percentile: None | tuple[float, float] = None,
    ):
        if taubin_smoothing:
            vo = torch.empty_like(surface.vertices)
            vo = vo.copy_(surface.vertices)

            surface.smooth_taubin(a=0.9, b=-0.95, inplace=True)

        K = surface.compute_laplace_beltrami_operator()
        H = surface.compute_mean_curvature(K)

        if H_clip_to_percentile:
            H.clamp_(
                *H.quantile(torch.tensor(H_clip_to_percentile, device=H.device))
            )

        if taubin_smoothing:
            surface.vertices = vo

        # weigh sampling probability by (abs) mean curvature
        # if curv_weight > 0:
        #     face_absH = surface.vertex_feature_to_face_feature(H.abs())
        #     weight = 1 + curv_weight * face_absH.clamp(
        #         face_absH.quantile(0.01), face_absH.quantile(0.99)
        #     )
        # else:
        #     weight = None

        weight = None

        points, sf, sb = surface.sample_points(
            n_samples,
            weights=weight,
            return_sampled_faces_and_bc=True,
        )
        sampled_K = surface.interpolate_vertex_feature_to_barycentric_coords(K, sf, sb)
        sampled_H = surface.interpolate_vertex_feature_to_barycentric_coords(H, sf, sb)
        if "weights" in surface.vertex_data:
            sampled_weights = surface.interpolate_vertex_feature_to_barycentric_coords(
                surface.vertex_data["weights"], sf, sb
            )
        else:
            sampled_weights = None
        return points, sampled_K, sampled_H, sampled_weights

    # def precompute_for_surface_loss(self, y_pred: dict, y_true: dict):
    #     index_name = "index"
    #     curvature_name = "curv"

    #     n_smooth = 1

    #     if self.has_SymmetricLoss:  # chamfer *or* curvature
    #         # compute nearest neighbors
    #         for h, surfaces in y_pred.items():
    #             for s in surfaces:
    #                 # these are indices into y_true!
    #                 index = y_pred[h][s].nearest_neighbor(y_true[h][s])
    #                 y_pred[h][s].vertex_data[index_name] = index
    #                 # these are indices into y_pred!
    #                 index = y_true[h][s].nearest_neighbor(y_pred[h][s])
    #                 y_true[h][s].vertex_data[index_name] = index

    #     # compute curvature
    #     if self.has_CurvatureLoss:
    #         for h, surfaces in y_pred.items():
    #             for s in surfaces:
    #                 curv = y_pred[h][s].compute_laplace_beltrami_operator()
    #                 y_pred[h][s].vertex_data[curvature_name] = curv

    #                 curv = y_true[h][s].compute_laplace_beltrami_operator()
    #                 if n_smooth > 0:
    #                     # smooth the true surface curvature
    #                     curv = y_true[h][s].compute_iterative_spatial_smoothing(
    #                         curv, n_smooth
    #                     )
    #                 y_true[h][s].vertex_data[curvature_name] = curv

    def forward(self, y_pred, y_true):
        """Compute all losses that is possible given the entries in `y_pred`"""

        # Compute raw loss
        loss_dict = {}
        for head, losses in self._active_losses.items():
            loss_dict[head] = {}
            for loss in losses:
                loss_fn = self.loss_functions[head][loss]
                # we try as this will usually be okay
                try:
                    match loss_fn:
                        case brainnet.modules.loss_wrappers.SupervisedLoss():
                            value = loss_fn(y_pred, y_true)
                        case brainnet.modules.loss_wrappers.RegularizationLoss():
                            value = loss_fn(y_pred)
                        case _:
                            raise ValueError
                    loss_dict[head][loss] = value
                except KeyError:
                    # Required data does not exist in y_pred and/or y_true
                    pass


        # loss_dict = {}
        # for head, head_losses in self.loss_functions.items():
        #     loss_dict[head] = {}
        #     found = False

        #     for name, loss in head_losses.items():

        #         if self._loss_weights[head][name] > 0.0:
        #             # we try as this will usually be okay
        #             try:
        #                 match loss:
        #                     case loss_wrappers.SupervisedLoss():
        #                         value = loss(y_pred, y_true)
        #                     case loss_wrappers.RegularizationLoss():
        #                         value = loss(y_pred)
        #                     case _:
        #                         raise ValueError
        #                 loss_dict[head][name] = value
        #                 found = True
        #             except KeyError:
        #                 # Required data does not exist in y_pred and/or y_true
        #                 pass

        #     if not found:
        #         del loss_dict[head]

        # set the across task normalizer

        # wn = 1 / sum(self.task_weights[t] for t in loss_dict)
        # self.across_task_normalizer = {
        #     t: self.task_weights[t] * wn for t in loss_dict
        # }

        return loss_dict


import brainnet.utilities

from typing import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

class CriterionAggregator(Metric):

    required_output_keys: tuple[str,str] = ("raw", "weighted") #("y_pred", "y", "criterion_kwargs")
    _state_dict_all_req_keys: tuple[str,str] = ("_sum", "_num_examples")

    def __init__(
        self,
        # loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        # batch_size: Callable = len,
        device: str | torch.device = torch.device("cpu"),
    ):
        """This "metric" is based on ignite.metrics.Loss but works with a dict of
        (averaged) losses rather than computing a single loss from y_pred and
        y. All entries (losses) are averaged separately.

        NOTE This is accurate only when all batches are of equal size!
        """
        super().__init__(output_transform, device=device)
        # self._loss_fn = loss_fn
        # self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = {}
        self._num_examples = {}

    @reinit__is_reduced
    def update(self, input_loss: tuple[dict, dict]) -> None:
        # if len(output) == 2:
        #     y_pred, y = cast(tuple[torch.Tensor, torch.Tensor], output)
        #     kwargs: dict = {}
        # else:
        #     y_pred, y, kwargs = cast(tuple[torch.Tensor, torch.Tensor, dict], output)

        # loss averaged over batch
        # loss = self._loss_fn(y_pred, y, **kwargs).detach()

        # the input is converted from mapping to tuple so convert back
        loss = dict(zip(self.required_output_keys, input_loss))

        # n = self._batch_size(y)
        batch_size = 1
        if batch_size > 1:
            brainnet.utilities.recursive_dict_multiply(loss, batch_size)
        brainnet.utilities.add_dict(self._sum, loss)
        brainnet.utilities.increment_dict_count(self._num_examples, loss, batch_size)


    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> dict:
        if len(self._num_examples) == 0:
            raise NotComputableError("Loss must have at least one example before it can be computed.")
        return brainnet.utilities.divide_dict(self._sum, self._num_examples)
