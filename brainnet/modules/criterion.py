import copy
from typing import Any, Callable, Mapping

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

from brainnet.config import LossParameters
import brainnet.modules.losses_surface
import brainnet.dict_utils


def recursive_dict_setter(d, k, v):
    if len(k) == 1:
        d[k[0]] = v
    else:
        return recursive_dict_setter(d[k[0]], k[1:], v)

class Criterion(torch.nn.Module):

    def __init__(self, config: LossParameters) -> None:
        super().__init__()

        self.loss_functions = config.functions

        self._head_weights = copy.deepcopy(config.head_weights)
        self._loss_weights = copy.deepcopy(config.loss_weights)
        self._set_active_heads() # sets everything

        self._state_attrs = ("_head_weights", "_loss_weights", "_active_heads", "_active_losses", "_needs_sampling")

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

    def state_dict(self):
        state_dict = super().state_dict()
        for attr in self._state_attrs:
            state_dict[attr] = getattr(self, attr)
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False
    ):
        for attr in self._state_attrs:
            if isinstance(getattr(self, attr), dict):
                brainnet.dict_utils.recursive_dict_update_(
                    getattr(self, attr), state_dict.pop(attr)
                )
            else:
                setattr(self, attr, state_dict.pop(attr))
        self._set_active_heads()

        super().load_state_dict(state_dict, strict, assign)

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

    @staticmethod
    def _is_class_instance(name, instance):
        """_summary_

        Get the class by converting its string representation

            <class '__main__.wrap_semi_hard_reduction.<locals>.SemiHardReduction'>

        to

            ['__main__', 'wrap_semi_hard_reduction', '<locals>', 'SemiHardReduction']

        Returns
        -------
        _type_
            _description_
        """
        return name == str(instance).strip("<>").strip("class").strip().strip("'").split(".")[-1]

    def _set_needs_sampling(self):
        """We keep track of this because we can avoid some calculations (e.g.,
        sampling points and finding nearest neighbors) when there is no active
        chamfer/curvature loss.
        """

        self._needs_sampling = any(
            self._is_class_instance("IndexMatchedData()", self.loss_functions[head][loss].loss_fn)
            for head, v in self._active_losses.items() for loss in v
        )
        self._needs_curvature = self._needs_sampling and any(
            self._is_class_instance("IndexMatchedData()", self.loss_functions[head][loss].loss_fn)  and self.loss_functions[head][loss].loss_fn.value_key == "sampled_H"
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
        # H_clip_to_percentile = dict(
        #     white = (0.001, 0.999),
        #     pial = (0.001, 0.999),
        # )

        H_clip_to_values = (-10, 10)

        for h, surfaces in y_pred.items():
            for s in surfaces:
                self._sample_points_curv_data(
                    y_pred[h][s],
                    n_samples,
                )
                self._sample_points_curv_data(
                    y_true[h][s],
                    n_samples,
                    smooth_y_true,
                    # H_clip_to_percentile = H_clip_to_percentile[s],
                    H_clip_to_values = H_clip_to_values,
                    # set_medial_wall_weights = True,
                )

                # these are indices into y_true!
                index = y_pred[h][s].nearest_neighbor_tensors(
                    y_pred[h][s].interpolated["points"],
                    y_true[h][s].interpolated["points"],
                )
                y_pred[h][s].interpolated["data"]["chamfer_index"] = index

                # =============================================================
                # NOTE
                # Since we do not know where a face/vertex is located exactly
                # on y_pred, set the interpolated data (e.g., medial wall
                # weights) of the sampled points on y_pred to the value of the
                # corresponding (closest) point on y_true
                for k in y_pred[h][s].vertex_data:
                    y_pred[h][s].interpolated["data"][k] = y_true[h][s].interpolated["data"][k].gather(-1, index)
                # =============================================================

                # these are indices into y_pred!
                index = y_true[h][s].nearest_neighbor_tensors(
                    y_true[h][s].interpolated["points"],
                    y_pred[h][s].interpolated["points"],
                )
                y_true[h][s].interpolated["data"]["chamfer_index"] = index


    def _sample_points_curv_data(
        self,
        surface,
        n_samples: int,
        taubin_smoothing: bool = False,
        # H_clip_to_percentile: None | tuple[float, float] = None,
        H_clip_to_values: None | tuple[float, float] = None,
    ):
        samp_p, samp_face, samp_coo = surface.sample_points(
            n_samples, return_sampled_faces_and_bc=True,
        )
        surface.interpolated["points"] = samp_p
        surface.interpolated["face_index"] = samp_face
        surface.interpolated["baricenter"] = samp_coo

        if self._needs_curvature:
            K = surface.compute_laplace_beltrami_operator()
            H = surface.compute_mean_curvature(K)
            H = surface.smooth_taubin(H) if taubin_smoothing else H

            if H_clip_to_values is not None:
                H.clamp_(*H_clip_to_values)

            surface.interpolated["data"]["K"] = surface.interpolate_vertex_features(K, samp_face, samp_coo)
            surface.interpolated["data"]["H"] = surface.interpolate_vertex_features(H, samp_face, samp_coo)

        for k,v in surface.vertex_data.items():
            surface.interpolated["data"][k] = surface.interpolate_vertex_features(
                v, samp_face, samp_coo
            )

    def forward(self, y_pred, y_true):
        """Compute all losses that is possible given the entries in `y_pred`"""

        # Compute raw loss
        loss_dict = {}
        for head, losses in self._active_losses.items():
            loss_dict[head] = {}
            for loss in losses:
                loss_fn = self.loss_functions[head][loss]
                # we try as this will usually be okay
                # try:
                match loss_fn:
                    case brainnet.modules.loss_wrappers.SupervisedLoss():
                        value = loss_fn(y_pred, y_true)
                    case brainnet.modules.loss_wrappers.RegularizationLoss():
                        value = loss_fn(y_pred)
                    case _:
                        raise ValueError
                loss_dict[head][loss] = value
                # except KeyError:
                #     # warnings.warn(f"Required data for {head}/{loss} does not exist in y_pred and/or y_true. Skipping.")
                #     pass

        return loss_dict


class CriterionAggregator(Metric):

    required_output_keys: tuple[str,str] = ("raw", "weighted") #("y_pred", "y", "criterion_kwargs")
    _state_dict_all_req_keys: tuple[str,str] = ("_sum", "_num_examples")

    def __init__(
        self,
        # loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        batch_size: Callable = len,
        device: str | torch.device = torch.device("cpu"),
    ):
        """This "metric" is based on ignite.metrics.Loss but works with a dict of
        (averaged) losses rather than computing a single loss from y_pred and
        y. All entries (losses) are averaged separately.
        """
        super().__init__(output_transform, device=device)
        # self._loss_fn = loss_fn
        self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = {}
        self._num_examples = {}

    @reinit__is_reduced
    def update(self, output: tuple) -> None:
        if len(output) == 4:
            loss, x, _, _ = output # out signature: loss, x, y_pred, y_true
        else:
            ValueError(f"Wrong output signature from engine for CriterionAggregator. Expected (loss, x, y_pred, y_true), got output of length {len(output)}.")

        # the input is converted from mapping to tuple so convert back
        # loss = dict(zip(self.required_output_keys, input_loss))

        batch_size = self._batch_size(x)
        if batch_size > 1:
            brainnet.dict_utils.recursive_dict_multiply(loss, batch_size)
        brainnet.dict_utils.add_dict(self._sum, loss)
        brainnet.dict_utils.increment_dict_count(self._num_examples, loss, batch_size)

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> dict:
        if len(self._num_examples) == 0:
            raise NotComputableError("Loss must have at least one example before it can be computed.")
        return brainnet.dict_utils.divide_dict(self._sum, self._num_examples)
