import copy
from typing import Any, Callable, Mapping

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

from brainnet.config import LossParameters
import brainnet.dict_utils
from brainnet.mesh.surface import Surface
from brainnet.utils import unsqueeze_and_expand


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
        self._set_active_heads()  # sets everything

        self._state_attrs = ("_head_weights", "_loss_weights")

    def state_dict(self):
        state_dict = super().state_dict()
        for attr in self._state_attrs:
            state_dict[attr] = getattr(self, attr)
        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        for attr in self._state_attrs:
            if isinstance(getattr(self, attr), dict):
                brainnet.dict_utils.recursive_dict_update_(
                    getattr(self, attr), state_dict.pop(attr)
                )
            else:
                setattr(self, attr, state_dict.pop(attr))
        self._set_active_heads()

        # del state_dict["_needs_sampling"]

        super().load_state_dict(state_dict, strict, assign)

    def _set_active_heads(self):
        self._active_heads = [h for h, v in self._head_weights.items() if v > 0.0]
        # if any heads changed status we need to update active losses
        self._set_active_losses()

    def _set_active_losses(self):
        # if a head is inactive, all its losses are ignored
        self._active_losses = {
            head: [n for n, v in self._loss_weights[head].items() if v > 0.0]
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
        return (
            name
            == str(instance)
            .strip("<>")
            .strip("class")
            .strip()
            .strip("'")
            .split(".")[-1]
        )

    def _set_needs_sampling(self):
        """We keep track of this because we can avoid some calculations (e.g.,
        sampling points and finding nearest neighbors) when there is no active
        chamfer/curvature loss.
        """
        self._needs_chamfer = any(
            self._is_class_instance(
                "IndexMatchedData()", self.loss_functions[head][loss].loss_fn
            )
            for head, v in self._active_losses.items()
            for loss in v
        )
        self._needs_sampling = any(
            self._is_class_instance(
                "IndexMatchedData()", self.loss_functions[head][loss].loss_fn
            )
            and self.loss_functions[head][loss].loss_fn.value_key[0] == "interpolated"
            for head, v in self._active_losses.items()
            for loss in v
        )
        self._needs_curvature = self._needs_sampling and any(
            self._is_class_instance(
                "IndexMatchedData()", self.loss_functions[head][loss].loss_fn
            )
            and self.loss_functions[head][loss].loss_fn.value_key[1] == "H"
            for head, v in self._active_losses.items()
            for loss in v
        )

    def update_head_weights(self, weights):
        for k, v in weights.items():
            if isinstance(k, (list, tuple)):
                recursive_dict_setter(self._head_weights, k, v)
            else:
                self._head_weights[k] = v
        self._set_active_heads()

    def update_loss_weights(self, weights):
        for k, v in weights.items():
            if isinstance(k, (list, tuple)):
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
                loss: value
                # / value.detach()
                * self._loss_weights[head][loss]
                * self._head_weights[head]
                for loss, value in losses.items()
            }
            for head, losses in loss_dict.items()
        }

    def _symmetrize_interpolated_data(self, y_pred, y_true):
        # For data that is only present in *y_true*, set the
        # interpolated data of the sampled points on y_pred to the
        # value of the corresponding (closest) point on y_true
        # (e.g., medial wall label)
        y_true_interp = y_true.interpolated.data.keys()
        y_pred_interp = y_pred.interpolated.data.keys()

        index = y_pred.interpolated.data["chamfer_index"]
        for k in y_true_interp:
            if k not in y_pred.interpolated.data:
                x = y_true.interpolated.data[k]
                index = unsqueeze_and_expand(index, x)
                y_pred.interpolated.data[k] = x.gather(-1, index)

        # For data that is only present in *y_pred*, set the
        # interpolated data of the sampled points on y_true to the
        # value of the corresponding (closest) point on y_pred
        # (e.g., uncertainty/sigma estimate from model)
        index = y_true.interpolated.data["chamfer_index"]
        for k in y_pred_interp:
            if k not in y_true.interpolated.data:
                x = y_pred.interpolated.data[k]
                index = unsqueeze_and_expand(index, x)
                y_true.interpolated.data[k] = x.gather(1, index)

        # return y_pred, y_true

    def _set_interpolated(self, y_to, y_from):
        """Set the interpolated state of `y_to` with the state of `y_from`.
        That is, use the interpolated points and interpolated data (e.g.,
        chamfer indices) estimated on one surface"""
        y_to.interpolated.face_index = y_from.interpolated.face_index
        y_to.interpolated.weights = y_from.interpolated.weights
        y_to.interpolated.points = y_to.interpolate_vertex_features(
            y_to.vertices, y_to.interpolated.face_index, y_to.interpolated.weights
        )

        y_to.interpolated.data |= y_from.interpolated.data

    def prepare_for_surface_loss(self, y_pred: dict, y_true: dict, smooth_y_true=True):
        """Precompute useful things for calculating the losses."""

        if not (self._needs_chamfer or self._needs_sampling):
            return

        for s, surfaces in y_pred.items():
            for h in surfaces:
                ypsh = y_pred[s][h]
                ytsh = y_true[s][h]
                if self._needs_sampling:
                    if s == "sphere.reg":
                        ypwh = y_pred["white"][h]
                        ytwh = y_true["white"][h]
                        self._set_interpolated(ypsh, ypwh)
                        self._set_interpolated(ytsh, ytwh)

                        # Add quantities for area and metric distortion losses
                        ypsh.face_data["original_area"] = ypwh.compute_face_areas()
                        ypsh.face_data["original_edge_norm"] = ypwh.compute_edge_norm()

                    else:
                        # implies that we require chamfer

                        self._sample_points_curv_data(ypsh)
                        self._sample_points_curv_data(
                            ytsh, taubin_smoothing=smooth_y_true
                        )

                        # NOTE these are indices into y_true!
                        index = ypsh.nearest_neighbor_tensors(
                            ypsh.interpolated.points,
                            ytsh.interpolated.points,
                        )
                        ypsh.interpolated.data["chamfer_index"] = index

                        # NOTE these are indices into y_pred!
                        index = ytsh.nearest_neighbor_tensors(
                            ytsh.interpolated.points,
                            ypsh.interpolated.points,
                        )
                        ytsh.interpolated.data["chamfer_index"] = index

                        self._symmetrize_interpolated_data(ypsh, ytsh)

                elif self._needs_chamfer:
                    index = ypsh.nearest_neighbor(ytsh)
                    ypsh.vertex_data["chamfer_index"] = index

                    index = ytsh.nearest_neighbor(ypsh)
                    ytsh.vertex_data["chamfer_index"] = index

    def _sample_points_curv_data(
        self,
        surface,
        n_samples: int = 100000,
        taubin_smoothing: bool = False,
        # H_clip_to_percentile: None | tuple[float, float] = None,
        H_clip_to_values: None | tuple[float, float] = None,
    ):
        _ = surface.sample_points(n_samples)

        # ss = Surface(
        #     surface.smooth_taubin(
        #         surface.smooth_gauss(surface.vertices, n_iter=2), n_iter=2
        #     ),
        #     surface.topology,
        # )
        # n = ss.compute_vertex_normals()
        # n = ss.interpolate_vertex_features(n, samp_face, samp_coo)
        # surface.interpolated.data["normal"] = n / n.norm(dim=-1, keepdim=True)

        if self._needs_curvature:
            if taubin_smoothing:
                ss = Surface(surface.smooth_taubin(), surface.topology)
            else:
                ss = surface

            K = ss.compute_laplace_beltrami_operator()
            H = ss.compute_mean_curvature(K)
            q = torch.tensor([0.005, 0.995], device=H.device)
            H = H.clamp(*H.quantile(q))

            # G = ss.compute_gaussian_curvature()
            # k1,k2 = ss.compute_principal_curvatures(H,G)
            # k1 = k1.clamp(min=-5.0, max=5.0)
            # k2 = k2.clamp(min=-5.0, max=5.0)

            # surface.vertex_data["H"] = H

            # surface.interpolated.data["K"] = surface.interpolate_vertex_features(
            #     K, samp_face, samp_coo
            # )
            surface.interpolated.data["H"] = surface.interpolate_vertex_features(
                H, surface.interpolated.face_index, surface.interpolated.weights
            )

        for k, v in surface.vertex_data.items():
            surface.interpolated.data[k] = surface.interpolate_vertex_features(
                v, surface.interpolated.face_index, surface.interpolated.weights
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
    required_output_keys: tuple[str, str] = (
        "raw",
        "weighted",
    )  # ("y_pred", "y", "criterion_kwargs")
    _state_dict_all_req_keys: tuple[str, str] = ("_sum", "_num_examples")

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
        loss, x = output[:2]  # out signature: loss, x, y_pred, y_true

        # if len(output) == 4:
        #     loss, x, _, _ = output  # out signature: loss, x, y_pred, y_true
        # else:
        #     raise ValueError(
        #         f"Wrong output signature from engine for CriterionAggregator. Expected (loss, x, y_pred, y_true), got output of length {len(output)}."
        #     )

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
            raise NotComputableError(
                "Loss must have at least one example before it can be computed."
            )
        return brainnet.dict_utils.divide_dict(self._sum, self._num_examples)
