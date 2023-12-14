import importlib
from types import ModuleType

import monai
import torch

import brainnet.modules.losses
from brainnet.modules.brainnet import BrainNet


def get_loss_function(name, modules: None | list[ModuleType] = None):
    # module search order is: torch, monai, local
    modules = (
        [torch.nn.modules.loss, monai.losses, brainnet.modules.losses]
        if modules is None
        else modules
    )
    if len(modules) == 0:
        raise ValueError("Module could not be found")
    try:
        return getattr(modules[0], name)
    except AttributeError:
        return get_loss_function(name, modules[1:])




def loss_function_from_string(fn_str):
    """

    'my.module.function' -> return function from my.module

    """
    split = fn_str.split(".")
    mod, fn = ".".join(split[:-1]), split[-1]
    mod = importlib.import_module(mod)
    return getattr(mod, fn)


# import yaml
# class IImageLoss(yaml.YAMLObject):
#     yaml_tag = "!ImageLoss"
#     def __init__(self, loss, y_pred, y_true):
#         self = ImageLoss(loss, y_pred, y_true)


class RegularizationLoss(torch.nn.Module):
    def __init__(self, loss_fn: str, y_pred: str) -> None:
        """_summary_

        Parameters
        ----------
        loss : str
            E.g., DiceLoss, CrossEntropyLoss, L1Loss
        y_pred : str
            _description_
        y_true : str
            _description_
        """
        super().__init__()

        self.y_pred = y_pred

        self.loss = loss_function_from_string(loss_fn)()

    def forward(self, y_pred):
        return self.loss(y_pred[self.y_pred])


class SupervisedLoss(torch.nn.Module):
    def __init__(self, loss_fn: str, y_pred: str, y_true: str) -> None:
        """_summary_

        Parameters
        ----------
        loss : str
            E.g., DiceLoss, CrossEntropyLoss, L1Loss
        y_pred : str
            _description_
        y_true : str
            _description_
        """
        super().__init__()

        self.y_pred = y_pred
        self.y_true = y_true

        self.loss = loss_function_from_string(loss_fn)()

    def forward(self, y_pred, y_true):
        return self.loss(y_pred[self.y_pred], y_true[self.y_true])



class ModelSupervisedLoss(SupervisedLoss):
    def __init__(
        self, loss: str, y_pred: str, y_true: str, model: dict, state_dict: str
    ) -> None:
        super().__init__(loss, y_pred, y_true)

        self.model = BrainNet(model["feature_extractor"], model["tasks"])
        # self.model.load_state_dict(torch.load(state_dict))

    def forward(self, y_pred, y_true):
        """Apply supervised model to `y_pred` before comparing with desired target."""
        return self.loss(self.model(y_pred[self.y_pred]), y_true[self.y_true])


class SoftMaskedSupervisedLoss(SupervisedLoss):
    def __init__(self, loss, y_pred, y_true, mask, background_channel) -> None:
        super().__init__(loss, y_pred, y_true)
        self.mask = mask
        self.background_channel = background_channel

    def forward(self, y_pred, y_true):
        mask = 1.0 - y_true[self.mask][:, self.background_channel]
        return self.loss(y_pred[self.y_pred], y_true[self.y_true], mask)




class SurfaceLossHandler:
    def extract_surface_data(self, data):
        return data["surface"]

    def average_over_hemispheres(self, loss_fn, y_pred, y_true=None):
        """Average loss over hemispheres."""
        loss = 0.0
        for hemi in y_pred:
            if y_true is None:
                loss += loss_fn(y_pred[hemi])
            else:
                loss += loss_fn(y_pred[hemi], y_true[hemi])
        return loss / len(y_pred)


class SurfaceRegularizationLoss(RegularizationLoss, SurfaceLossHandler):
    def __init__(self, loss_fn: str, y_pred: str) -> None:
        super().__init__(loss_fn, y_pred)

    def forward(self, y_pred):
        return self.average_over_hemispheres(
            super().forward,
            self.extract_surface_data(y_pred),
        )


class SurfaceSupervisedLoss(SupervisedLoss, SurfaceLossHandler):
    def __init__(self, loss_fn: str, y_pred: str, y_true: str) -> None:
        super().__init__(loss_fn, y_pred, y_true)

    def forward(self, y_pred, y_true):
        return self.average_over_hemispheres(
            super().forward,
            self.extract_surface_data(y_pred),
            self.extract_surface_data(y_true),
        )



class IndexedSupervisedLoss(torch.nn.Module):
    def __init__(self, loss_fn: str, y_pred: str, y_true: str) -> None:
        """_summary_

        Parameters
        ----------
        loss : str
            E.g., DiceLoss, CrossEntropyLoss, L1Loss
        y_pred : str
            _description_
        y_true : str
            _description_
        """
        super().__init__()

        self.y_pred = y_pred
        self.y_true = y_true

        self.loss = loss_function_from_string(loss_fn)()

    def forward(self, y_pred, y_true, i_pred, i_true):
        return self.loss(y_pred[self.y_pred][i_pred], y_true[self.y_true][i_true])


class OriginalSurfaceLoss(torch.nn.Module):
    def __init__(self, loss, y_pred, y_true) -> None:
        super().__init__()
        self.y_pred = y_pred
        self.y_true = y_true

        symloss = {"SymmetricChamferLoss", "SymmetricCurvatureLoss"}

        self.loss = {}
        for m in loss:
            if m in symloss:
                self.loss[m] = get_loss_function("SymmetricMSELoss")()
            else:
                self.loss[m] = get_loss_function(m)()

    def _compute_curvature(self, y_pred, y_true, y_true_curv=None):
        y_pred_curv = y_pred.compute_mean_curvature_vector()
        if y_true_curv is None:
            y_true_curv = y_true.compute_mean_curvature_vector()
            y_true_curv = y_true.compute_iterative_spatial_smoothing(y_true_curv, 3)
        return y_pred_curv, y_true_curv

    def _compute_neighbors(self, x, y, x_nn=None, y_nn=None):
        if x_nn is None:
            x_nn = y.nearest_neighbor(x)
        if y_nn is None:
            y_nn = x.nearest_neighbor(y)
        return x_nn, y_nn

    def forward(self, y_pred, y_true, curv_true: None | dict = None):
        x = y_pred[self.y_pred]
        y = y_true[self.y_true]

        x_nn, y_nn = None, None

        loss = {}
        for k, fn in self.loss.items():
            match k:
                case "MatchedDistanceLoss":
                    loss[k] = fn(x, y)
                case "SymmetricChamferLoss":
                    x_nn, y_nn = self._compute_neighbors(x, y, x_nn, y_nn)
                    loss[k] = fn(x, y, x_nn, y_nn)
                case "SymmetricCurvatureLoss":
                    x_nn, y_nn = self._compute_neighbors(x, y, x_nn, y_nn)
                    x_curv, y_curv = self._compute_curvature(x, y, curv_true)
                    loss[k] = fn(x_curv, y_curv, x_nn, y_nn)
                case "EdgeLengthVarianceLoss":
                    loss[k] = fn(x)

        # if (k := "MatchedDistanceLoss") in self.loss:
        #     loss[k] = self.loss[k](x, y)

        # if "SymmetricChamferLoss" in self.loss or "SymmetricCurvatureLoss" in self.loss:
        #     y_nn = x.nearest_neighbor(y)
        #     x_nn = y.nearest_neighbor(x)
        #     if (k := "SymmetricChamferLoss") in self.loss:
        #         loss[k] = self.loss[k](x, y, x_nn, y_nn)
        #     if (k := "SymmetricCurvatureLoss") in self.loss:
        #         x_curv = x.compute_mean_curvature_vector()
        #         if curv_true is None:
        #             y_curv = y.compute_mean_curvature_vector()
        #             y_curv = y.compute_iterative_spatial_smoothing(y_curv, 3)
        #         else:
        #             y_curv = curv_true
        #         loss[k] = self.loss[k](x_curv, y_curv, x_nn, y_nn)

        # if (k := "EdgeLengthVarianceLoss") in self.loss:
        #     loss[k] = self.loss[k](x)

        return loss
