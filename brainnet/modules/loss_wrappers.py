from types import ModuleType

import monai
import torch

import brainnet.modules.losses
from brainnet.modules.brainnet import BrainNet

def get_loss_function(name, modules: None | list[ModuleType] = None):
    # module search order is: torch, monai, local
    modules = [torch.nn.modules.loss, monai.losses, brainnet.modules.losses] if modules is None else modules
    if len(modules) == 0:
        raise ValueError("Module could not be found")
    try:
        return getattr(modules[0], name)
    except AttributeError:
        return get_loss_function(name, modules[1:])


class ImageLoss(torch.nn.Module):
    def __init__(self, loss: str, y_pred: str, y_true: str) -> None:
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

        self.loss = get_loss_function(loss)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred[self.y_pred], y_true[self.y_true])


class SupervisedImageLoss(ImageLoss):
    def __init__(self, loss: str, y_pred: str, y_true: str, model: dict, state_dict: str) -> None:
        super().__init__(loss, y_pred, y_true)

        self.model = BrainNet(model["feature_extractor"], model["tasks"])
        # self.model.load_state_dict(torch.load(state_dict))

    def forward(self, y_pred, y_true):
        """Apply supervised model to `y_pred` before comparing with desired target."""
        return self.loss(self.model(y_pred[self.y_pred]), y_true[self.y_true])


class SoftMaskedImageLoss(ImageLoss):
    def __init__(self, loss, y_pred, y_true, mask, background_channel) -> None:
        super().__init__(loss, y_pred, y_true)
        self.mask = mask
        self.background_channel = background_channel

    def forward(self, y_pred, y_true):
        mask = 1.0 - y_true[self.mask][:, self.background_channel]
        return self.loss(y_pred[self.y_pred], y_true[self.y_true], mask)


class SurfaceLoss(torch.nn.Module):
    def __init__(self, loss, y_pred, y_true) -> None:
        super().__init__()
        self.y_pred = y_pred
        self.y_true = y_true

        symloss = {"SymmetricChamferLoss", "SymmetricCurvatureLoss"}

        self.loss = {}
        for m in loss:
            if m in symloss:
                self.loss[m] = get_loss_function("SymmetricMSELoss")
            else:
                self.loss[m] = get_loss_function(m)

    def forward(self, y_pred, y_true, curv_true: None | dict = None):
        x = y_pred[self.y_pred]
        y = y_true[self.y_true]

        loss = {}

        if (k := "MatchedDistanceLoss") in self.loss:
            loss[k] = self.loss[k](x, y)

        if "SymmetricChamferLoss" in self.loss or "SymmetricCurvatureLoss" in self.loss:
            y_nn = x.nearest_neighbor(y)
            x_nn = y.nearest_neighbor(x)
            if (k := "SymmetricChamferLoss") in self.loss:
                loss[k] = self.loss[k](x, y, x_nn, y_nn)
            if (k := "SymmetricCurvatureLoss") in self.loss:
                x_curv = x.compute_mean_curvature_vector()
                if curv_true is None:
                    y_curv = y.compute_mean_curvature_vector()
                    y_curv = y.compute_iterative_spatial_smoothing(y_curv, 3)
                else:
                    y_curv = curv_true
                loss[k] = self.loss[k](x_curv, y_curv, x_nn, y_nn)

        if (k := "EdgeLengthVarianceLoss") in self.loss:
            loss[k] = self.loss[k](x)

        return loss
