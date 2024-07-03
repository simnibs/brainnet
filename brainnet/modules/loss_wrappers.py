import importlib

import torch

# from brainnet.modules.brainnet import BrainNet


def function_from_string(fn_str):
    """Get a function from its full string specification, e.g.,

        'my.module.function' -> return function from my.module

    """
    split = fn_str.split(".")
    mod, fn = ".".join(split[:-1]), split[-1]
    mod = importlib.import_module(mod)
    return getattr(mod, fn)


class RegularizationLoss(torch.nn.Module):
    def __init__(self, loss_fn, y_pred: None | str) -> None:
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

        # self.loss_fn = function_from_string(loss_fn)(**(loss_fn_kwargs or {}))
        self.loss_fn = loss_fn
        self.y_pred = y_pred


    def forward(self, y_pred):
        return self.loss_fn(
            y_pred if self.y_pred is None else y_pred[self.y_pred]
        )


class SupervisedLoss(torch.nn.Module):
    def __init__(self, loss_fn, y_pred: str | None, y_true: str | None) -> None:
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

        # self.loss_fn = function_from_string(loss_fn)(**(loss_fn_kwargs or {}))
        self.loss_fn = loss_fn
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self, y_pred, y_true, **kwargs):
        return self.loss_fn(
            y_pred if self.y_pred is None else y_pred[self.y_pred],
            y_true if self.y_true is None else y_true[self.y_true],
            **kwargs,
        )


# class ModelSupervisedLoss(SupervisedLoss):
#     def __init__(
#         self, loss_fn: str, y_pred: str, y_true: str, model: dict, state_dict: str, loss_fn_kwargs: dict | None = None
#     ) -> None:
#         super().__init__(loss_fn, y_pred, y_true, loss_fn_kwargs)

#         self.model = BrainNet(model["body"], model["heads"])
#         # self.model.load_state_dict(torch.load(state_dict))

#     def forward(self, y_pred, y_true):
#         """Apply supervised model to `y_pred` before comparing with desired target."""
#         return self.loss_fn(self.model(y_pred[self.y_pred]), y_true[self.y_true])


class SoftMaskedSupervisedLoss(SupervisedLoss):
    def __init__(self, loss_fn, y_pred: str, y_true: str, mask, background_channel) -> None:
        super().__init__(loss_fn, y_pred, y_true)
        self.mask = mask
        self.background_channel = background_channel

    def forward(self, y_pred, y_true):
        mask = 1.0 - y_true[self.mask][:, self.background_channel]
        return self.loss_fn(y_pred[self.y_pred], y_true[self.y_true], mask)


class SurfaceLossHandler:
    @staticmethod
    def extract_surface_data(data):
        return data["surface"]

    def average_over_hemispheres(self, loss_fn, y_pred, y_true=None, **kwargs):
        """Average loss over hemispheres."""
        yp = self.extract_surface_data(y_pred)
        yt = self.extract_surface_data(y_true) if y_true is not None else None
        loss = 0.0
        for hemi in yp:
            if yt is None:
                loss += loss_fn(yp[hemi], **kwargs)
            else:
                loss += loss_fn(yp[hemi], yt[hemi], **kwargs)
        return loss / len(yp)


class SurfaceRegularizationLoss(RegularizationLoss, SurfaceLossHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred, **kwargs):
        return self.average_over_hemispheres(super().forward, y_pred, **kwargs)


class SurfaceSupervisedLoss(SupervisedLoss, SurfaceLossHandler):
    def __init__(self, *args, **kwargs) -> None:
        """Extract surface data and calculate average loss over hemispheres."""
        super().__init__(*args, **kwargs) # init of SupervisedLoss!

    def forward(self, y_pred, y_true, **kwargs):
        return self.average_over_hemispheres(
            super().forward, y_pred, y_true, **kwargs,
        )