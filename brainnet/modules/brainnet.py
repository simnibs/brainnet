
from monai.networks import nets
import torch

from brainnet.modules import tasks


VALID_FEATURE_EXTRACTORS = {
    "UNet", "ResNet",
}


class BrainNet(torch.nn.Module):
    def __init__(self, feature_extractor_config, task_config):
        """Construct the BrainNet model consisting of feature extractor and
        tasks.

        Parameters
        ----------
        feature_extractor_config : _type_
            _description_
        task_config : _type_
            _description_
        """
        super().__init__()
        assert feature_extractor_config["model"] in VALID_FEATURE_EXTRACTORS, f"Invalid feature network. Please chose from {VALID_FEATURE_EXTRACTORS}"

        self.config = task_config

        # Build feature extractor
        self.feature_extractor = getattr(
            nets,
            feature_extractor_config["model"]
        )(**feature_extractor_config["kwargs"])

        # set `in_channels` for tasks to `out_channels` from feature extractor
        fe_out_ch = feature_extractor_config["kwargs"]["out_channels"]
        for t in task_config.values():
            if (k := "module_kwargs") not in t:
                t[k] = {}
            if (k := "in_channels") not in t:
                t[k] = fe_out_ch

        # Build tasks, e.g., segmentation, SR, surface
        self.tasks = torch.nn.ModuleDict(
            {n: getattr(tasks, t["module"])(**t["module_kwargs"]) for n,t in task_config.items()}
        )

        # Register a forward hook for the desired activation.
        if "BiasFieldNet" in self.tasks and self:

            # check that tasks[...].feature_submodule is not None...

            self._feature_activations = {}
            def _activation_from_layer(name):
                def hook(model, x, y):
                    self._feature_activations[name] = y
                return hook

            # string, e.g., model.1.submodule.1.submodule.1
            self._feature_hook = self.feature_extractor.get_submodule(
                task_config["feature_submodule"]
            ).register_forward_hook(_activation_from_layer("bias_field_features"))

    def forward(self, image: torch.Tensor, vertices: torch.Tensor | None = None) -> dict:
        """

        Parameters
        ----------
        image : torch.Tensor
            The image to process.

        Returns
        -------
        pred : dict
            Dictionary containing the output of each task using the task name
            as the key.
        """
        features = self.feature_extractor(image)
        pred = {}
        for name,task in self.tasks.items():
            if isinstance(task, tasks.TaskModule):
                y = task(features)
            elif isinstance(task, tasks.BiasFieldModule):
                if task.feature_submodule is None:
                    y = task(features)
                else:
                    intermediate_features = self._feature_activations["bias_field_features"]
                    y = task(intermediate_features)
                    self._feature_hook.remove()
            elif isinstance(task, tasks.ContrastiveModule):
                y = task(features)
            elif isinstance(task, tasks.SurfaceModule):
                y = task(features, vertices)
            else:
                raise ValueError(f"Unknown task class {task}")
            pred[name] = y
        return pred
