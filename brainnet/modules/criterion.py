import torch

from brainnet.modules import loss_wrappers
from brainnet.modules.losses import SymmetricDistanceLoss, SymmetricCurvatureLoss


class Criterion(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # self.weight_threshold = weight_threshold

        self.weights = vars(config.weights)

        self.losses = {
            name: self.setup_loss(loss_config)
            for name, loss_config in vars(config.functions).items()
        }

        self.has_SymmetricMSELoss = any(isinstance(v.loss_fn, SymmetricDistanceLoss) for v in self.losses.values())
        self.has_SymmetricCurvatureLoss = any(isinstance(v.loss_fn, SymmetricCurvatureLoss) for v in self.losses.values())

        self._weight_normalizer = 1.0

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


    def apply_normalized_weights(self, loss_dict):
        """Apply normalized weights. `forward` needs to be run in order to
        update the weight normalizer.
        """
        return {
            k: v * self.weights[k] * self._weight_normalizer for k, v in loss_dict.items()
        }

    def precompute_for_surface_loss(self, y_pred: dict, y_true: dict):
        """Precompute useful things for calculating the losses."""

        index_name = "index"
        curvature_name = "curv"

        if self.has_SymmetricMSELoss: # chamfer *or* curvature
            # compute nearest neighbors
            for h,surfaces in y_pred.items():
                for s in surfaces:
                    # these are indices into y_true!
                    index = y_pred[h][s].nearest_neighbor(y_true[h][s])
                    y_pred[h][s].vertex_data[index_name] = index
                    # these are indices into y_pred!
                    index = y_true[h][s].nearest_neighbor(y_pred[h][s])
                    y_true[h][s].vertex_data[index_name] = index

            # compute curvature
            if self.has_SymmetricCurvatureLoss:
                for h,surfaces in y_pred.items():
                    for s in surfaces:
                        y_pred[h][s].vertex_data[curvature_name] = y_pred[h][s].compute_mean_curvature_vector()
                        y_true[h][s].vertex_data[curvature_name] = y_true[h][s].compute_mean_curvature_vector()


    def forward(self, y_pred, y_true):
        """Compute all losses that is possible given the entries in `y_pred`"""
        weight_total = 0.0

        # Compute raw loss
        loss_dict = {}
        for name, loss in self.losses.items():
            try:
                match loss:
                    case loss_wrappers.SupervisedLoss():
                        # if isinstance(loss, brainnet.modules.losses.SymmetricMSELoss):
                        #     i_pred = loss.loss_fn.i_pred
                        #     i_true = loss.loss_fn.i_true
                        #     curv_true = kwargs["curv_true"]
                        value = loss(y_pred, y_true)
                    case loss_wrappers.RegularizationLoss():
                        value = loss(y_pred)
                    case _:
                        raise ValueError
                loss_dict[name] = value
                weight_total += self.weights[name]
            except KeyError:
                # Required data does not exist in y_pred and/or y_true
                pass

        # Compute weighted loss
        self._weight_normalizer = 1 / weight_total

        return loss_dict
