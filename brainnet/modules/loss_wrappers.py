import importlib

import torch

from brainnet.modules.brainnet import BrainNet


def function_from_string(fn_str):
    """Get a function from its full string specification, e.g.,

        'my.module.function' -> return function from my.module

    """
    split = fn_str.split(".")
    mod, fn = ".".join(split[:-1]), split[-1]
    mod = importlib.import_module(mod)
    return getattr(mod, fn)


class RegularizationLoss(torch.nn.Module):
    def __init__(self, loss_fn: str, y_pred: str, loss_fn_kwargs: dict | None = None) -> None:
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

        self.loss_fn = function_from_string(loss_fn)(**(loss_fn_kwargs or {}))

    def forward(self, y_pred):
        return self.loss_fn(y_pred[self.y_pred])


class SupervisedLoss(torch.nn.Module):
    def __init__(self, loss_fn: str, y_pred: str, y_true: str, loss_fn_kwargs: dict | None = None) -> None:
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

        self.loss_fn = function_from_string(loss_fn)(**(loss_fn_kwargs or {}))

    def forward(self, y_pred, y_true, **kwargs):
        return self.loss_fn(y_pred[self.y_pred], y_true[self.y_true], **kwargs)

class ModelSupervisedLoss(SupervisedLoss):
    def __init__(
        self, loss_fn: str, y_pred: str, y_true: str, model: dict, state_dict: str, loss_fn_kwargs: dict | None = None
    ) -> None:
        super().__init__(loss_fn, y_pred, y_true, loss_fn_kwargs)

        self.model = BrainNet(model["feature_extractor"], model["tasks"])
        # self.model.load_state_dict(torch.load(state_dict))

    def forward(self, y_pred, y_true):
        """Apply supervised model to `y_pred` before comparing with desired target."""
        return self.loss_fn(self.model(y_pred[self.y_pred]), y_true[self.y_true])


class SoftMaskedSupervisedLoss(SupervisedLoss):
    def __init__(self, loss_fn, y_pred, y_true, mask, background_channel, loss_fn_kwargs: dict | None = None) -> None:
        super().__init__(loss_fn, y_pred, y_true, loss_fn_kwargs)
        self.mask = mask
        self.background_channel = background_channel

    def forward(self, y_pred, y_true):
        mask = 1.0 - y_true[self.mask][:, self.background_channel]
        return self.loss_fn(y_pred[self.y_pred], y_true[self.y_true], mask)


class SurfaceLossHandler:
    def extract_surface_data(self, data):
        return data["surface"]

    def average_over_hemispheres(self, loss_fn, y_pred, y_true=None, **kwargs):
        """Average loss over hemispheres."""
        loss = 0.0
        for hemi in y_pred:
            if y_true is None:
                loss += loss_fn(y_pred[hemi], **kwargs)
            else:
                loss += loss_fn(y_pred[hemi], y_true[hemi], **kwargs)
        return loss / len(y_pred)


    # def average_over_hemispheres1(self, loss_fn, y_pred, y_true=None, **kwargs):
    #     """Average loss over hemispheres."""
    #     loss = 0.0
    #     for hemi in y_pred:
    #         if y_true is None:
    #             loss += loss_fn(y_pred[hemi].vertices, **kwargs)
    #         else:
    #             loss += loss_fn(y_pred[hemi].vertices, y_true[hemi].vertices, **kwargs)
    #     return loss / len(y_pred)

    # def average_over_hemispheres2(self, loss_fn, y_pred, y_true, i_pred=None, i_true=None):
    #     """Average loss over hemispheres."""
    #     self.i_pred = i_pred or {hemi: None for hemi in y_pred}
    #     self.i_true = i_true or {hemi: None for hemi in y_pred}
    #     loss = 0.0
    #     for hemi in y_pred:
    #         loss += loss_fn(
    #             y_pred[hemi].vertices,
    #             y_true[hemi].vertices,
    #             self.i_pred[hemi],
    #             self.i_true[hemi],
    #         )
    #     return loss / len(y_pred)

    # def average_over_hemispheres3(
    #         self,
    #         loss_fn,
    #         y_pred,
    #         curv_true: dict,
    #         i_pred=None,
    #         i_true=None,
    #     ):
    #     """Average loss over hemispheres."""
    #     self.i_pred = i_pred or {hemi: None for hemi in y_pred}
    #     self.i_true = i_true or {hemi: None for hemi in y_pred}
    #     loss = 0.0
    #     for hemi in y_pred:
    #         curv_pred = y_pred[hemi].compute_mean_curvature_vector()

    #         loss += loss_fn(
    #             curv_pred,
    #             curv_true[hemi],
    #             self.i_pred[hemi],
    #             self.i_true[hemi],

    #         )
    #     return loss / len(y_pred)

class SurfaceRegularizationLoss(RegularizationLoss, SurfaceLossHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred, **kwargs):
        return self.average_over_hemispheres(
            super().forward,
            self.extract_surface_data(y_pred),
            **kwargs,
        )


class SurfaceSupervisedLoss(SupervisedLoss, SurfaceLossHandler):
    def __init__(self, *args, **kwargs) -> None:
        """Extract surface data and calculate average loss over hemispheres."""
        super().__init__(*args, **kwargs) # init of SupervisedLoss!

    def forward(self, y_pred, y_true, **kwargs):
        return self.average_over_hemispheres(
            super().forward,
            self.extract_surface_data(y_pred),
            self.extract_surface_data(y_true),
            **kwargs,
        )



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
