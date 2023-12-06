from copy import deepcopy

import torch

import brainnet.utilities
import brainnet.modules.loss_wrappers

# class Criterion(torch.nn.Module):
#     def __init__(self, config_loss_weight, config_task_weight, device) -> None:
#         """_summary_

#         Parameters
#         ----------
#         task_loss : _type_
#             _description_
#         task_weight : _type_
#             _description_
#         """
#         super().__init__()
#         tasks = [task for task in config_loss_weight]
#         self.loss_tags = [(task, loss) for task,losses in config_task_weight.items() for loss in losses]
#         self.n_tasks = len(tasks)
#         self.n_losses = len(self.loss_tags)

#         self.loss = {
#             task: {
#                 loss_class: getattr(MedID_losses, loss_class)() for loss_class in losses
#             } for task,losses in config_task_weight.items()
#         }

#         self.loss_weight = torch.zeros(self.n_losses, device=device)
#         self.task_weight = torch.zeros(self.n_tasks, device=device)

#     def update_loss_and_task_weight(self):
#         i = 0
#         j = 0
#         for task,losses in config_task_weight.items():
#             self.task_weight[i] = config_task_weight[task]
#             i += 1
#             for loss_class, loss_val in losses.items():
#                 self.loss_weight[j] = loss_val
#                 j += 1

#         self.total_loss_weight = self.loss_weight.sum()
#         self.total_task_weight = self.task_weight.sum()





#         self.has_chamfer_loss = any(
#             isinstance(MedID_losses.SymmetricChamferLoss, tuple(v for v in self.loss.values()))
#         )
#         self.has_curvature_loss = any(
#             isinstance(MedID_losses.SymmetricCurvatureLoss, tuple(v for v in self.loss.values()))
#         )


#     def apply_weights(self, loss):
#         """Apply weights (across and within tasks)."""
#         return {k: loss[k] * self.weights[k] for k in loss}

#     def sum(self, loss, apply_weights=False):
#         loss = self.apply_weights(loss) if apply_weights else loss
#         return torch.sum(v for v in loss.values())


#     def forward(self, y_pred, y_true):
#         """Compute losses

#         Parameters
#         ----------
#         y_pred : dict
#         y_true : dict


#         Returns
#         -------

#         """
#         loss = torch.zeros(self.n_losses)

#         for task in y_pred:
#             for name,loss_class in self.loss[task]:
#                 if isinstance(loss_class, self.regularization):
#                     loss[i] = loss_class(y_pred[task])
#                 else:
#                     loss[i] = loss_class(y_pred[task], y_true[task])


class Criterion(torch.nn.Module):
    def __init__(self, loss_config, weights_config) -> None:
        super().__init__()

        self.weight_threshold = 1e-8

        self._original_weights = weights_config
        self.weights = deepcopy(weights_config)
        self._normalize_weights()

        self.losses = {
            task: {name: self.get_loss(loss, task) for name,loss in losses.items()}
            for task, losses in loss_config.items()
        }

    @staticmethod
    def get_loss(kwargs, task):
        assert "module" in kwargs, "Loss definition should contain `module` definition"
        assert "loss" in kwargs, "Loss definition should contain `loss` definition"
        module = kwargs["module"]
        kwargs_ = {k:v for k,v in kwargs.items() if k != "module"}
        if "y_pred" not in kwargs_:
            kwargs_["y_pred"] = task
        if "y_true" not in kwargs:
            kwargs_["y_true"] = task
        return getattr(brainnet.modules.loss_wrappers, module)(**kwargs_)

    def update_weights(self, weights):
        # we need the copy because self.weights are normalized and weights are not
        self.weights = deepcopy(self._original_weights)
        brainnet.utilities.recursive_dict_update_(self.weights, weights)
        self._normalize_weights()

    def _normalize_weights(self):
        total = brainnet.utilities.recursive_dict_sum(self.weights)
        brainnet.utilities.recursive_dict_multiply(self.weights, 1/total)

    @staticmethod
    def flatten_loss(loss):
        return brainnet.utilities.flatten_dict(loss)

    def apply_weights(self, loss):
        """"""
        return brainnet.utilities.multiply_dicts(loss, self.weights)

    def forward(self, y_pred, y_true):
        """Compute all losses."""
        out = {}
        for task, losses in self.losses.items():
            for name,loss in losses.items():
                if isinstance(loss, dict):
                    out[task][name] = {
                        k: v(y_pred, y_true) if (self.weights[task][name][k] > self.weight_threshold) else 0 for k,v in loss.items()
                    }
                else:
                    out[task][name] = loss(y_pred, y_true) if (self.weights[task][name] > self.weight_threshold) else 0
        return out

    # @staticmethod
    # def _forward(y_pred, y_true, losses, weights, out):
    #     if out is None:
    #         out = {}
    #     for k,v in losses.items():
    #         if isinstance(v, dict):
    #             out[k] = {}
    #             self._forward(y_pred, y_true, out[k], )
    #         else:
    #             out[k] = loss(y_pred, y_true) if self.weights[task][name] > self.weight_threshold else 0
    #     return out
