import torch

from brainnet.modules import loss_wrappers
from brainnet.modules import losses
from brainsynth.config.utilities import recursive_namespace_to_dict

def recursive_dict_setter(d, k, v):
    if len(k) == 1:
        d[k[0]] = v
    else:
        return recursive_dict_setter(d[k[0]], k[1:], v)

class Criterion(torch.nn.Module):

    def _update_has_ChamferLoss(self):
        self.has_ChamferLoss = any(
            isinstance(
                loss.loss_fn,
                (
                    losses.AsymmetricChamferLoss,
                    losses.SymmetricChamferLoss,
                    losses.SymmetricCurvatureNormLoss,
                    losses.AsymmetricCurvatureNormLoss,
                ),
            )
            and (self.head_weights[head] > 0.0)
            and (self.loss_weights[head][name] > 0.0)
            for head, head_losses in self.losses.items()
            for name, loss in head_losses.items()
        )

    def _update_active_losses(self):
        self.active_losses = {
            head: [name for name in losses if self.loss_weights[head][name] > 0.0]
            for head, losses in self.losses.items()
        }

    @property
    def loss_weights(self):
        return self._loss_weights

    @loss_weights.setter
    def loss_weights(self, value):

        if hasattr(self, "_loss_weights"):
            for k,v in value.items():
                if isinstance(k, (list,tuple)):
                    recursive_dict_setter(self._loss_weights, k, v)
                else:
                    self._loss_weights[k] = v
        else:
            self._loss_weights = value

        self._update_has_ChamferLoss()
        self._update_active_losses()


    @property
    def head_weights(self):
        return self._head_weights

    @head_weights.setter
    def head_weights(self, value):
        if hasattr(self, "_head_weights"):
            for k,v in value.items():
                if isinstance(k, (list,tuple)):
                    recursive_dict_setter(self._head_weights, k, v)
                else:
                    self._head_weights[k] = v
        else:
            self._head_weights = value


        self._update_has_ChamferLoss()

    def __init__(self, config) -> None:
        super().__init__()

        # self.weight_threshold = weight_threshold
        self.config = config

        self.head_weights = recursive_namespace_to_dict(config.head_weights)
        self.loss_weights = recursive_namespace_to_dict(config.loss_weights)

        # across_task_normalizer is computed on every forward pass depending on
        # which task losses are feasible
        # self.within_task_normalizer = {
        #     task: 1 / sum(w for w in losses.values())
        #     for task, losses in self.loss_weights.items()
        # }
        # self.across_task_normalizer = {}

        self.losses = {
            head: {
                loss_name: self.setup_loss(loss_config)
                for loss_name, loss_config in vars(head_losses).items()
            }
            for head, head_losses in vars(config.functions).items()
        }

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

        # self.has_SymmetricLoss = any(
        #     isinstance(
        #         loss.loss_fn, (losses.SymmetricMeanSquaredNormLoss, losses.ASymmetricCurvatureNormLoss)
        #     )
        #     for task_losses in self.losses.values()
        #     for loss in task_losses.values()
        # )
        # self.has_CurvatureLoss = any(
        #     isinstance(loss.loss_fn, (losses.SymmetricCurvatureNormLoss, losses.MatchedCurvatureLoss, losses.AsymmetricCurvatureAngleLoss))
        #     for task_losses in self.losses.values()
        #     for loss in task_losses.values()
        # )
        self.has_ChamferLoss = any(
            isinstance(
                loss.loss_fn,
                (
                    losses.AsymmetricChamferLoss,
                    losses.SymmetricChamferLoss,
                    losses.SymmetricCurvatureNormLoss,
                    losses.AsymmetricCurvatureNormLoss,
                ),
            )
            and (self.head_weights[head] > 0.0)
            and (self.loss_weights[head][name] > 0.0)
            for head, head_losses in self.losses.items()
            for name, loss in head_losses.items()
        )

        self.active_losses = {
            head: [name for name in losses if self.loss_weights[head][name] > 0.0]
            for head, losses in self.losses.items()
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

    def apply_weights(self, loss_dict):
        """Apply normalized weights. `forward` needs to be run in order to
        update the weight normalizer.

        Each individual loss is weighted as

            loss * loss_weight * within_task_normalizer * across_task_normalizer

        such the loss weights within and across tasks sum to one.
        """
        return {
            head: {
                name: loss * self.loss_weights[head][name]
                # * self.within_task_normalizer[task]
                * self.head_weights[head]
                # * self.across_task_normalizer[task]
                for name, loss in losses.items()
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

        if not self.has_ChamferLoss:
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
        for head, head_losses in self.losses.items():
            loss_dict[head] = {}
            found = False

            for name, loss in head_losses.items():

                if self.loss_weights[head][name] > 0.0:
                    # we try as this will usually be okay
                    try:
                        match loss:
                            case loss_wrappers.SupervisedLoss():
                                value = loss(y_pred, y_true)
                            case loss_wrappers.RegularizationLoss():
                                value = loss(y_pred)
                            case _:
                                raise ValueError
                        loss_dict[head][name] = value
                        found = True
                    except KeyError:
                        # Required data does not exist in y_pred and/or y_true
                        pass

            if not found:
                del loss_dict[head]

        # set the across task normalizer

        # wn = 1 / sum(self.task_weights[t] for t in loss_dict)
        # self.across_task_normalizer = {
        #     t: self.task_weights[t] * wn for t in loss_dict
        # }

        return loss_dict
