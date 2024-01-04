from types import SimpleNamespace

from monai.networks import nets
import torch

from brainnet.modules import tasks

VALID_FEATURE_EXTRACTORS = {
    "UNet",
    "ResNet",
}


class BrainNet(torch.nn.Module):
    def __init__(self, feature_extractor_config, task_config, device):
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

        device = torch.device(device)

        assert (
            feature_extractor_config.model in VALID_FEATURE_EXTRACTORS
        ), f"Invalid feature network. Please chose from {VALID_FEATURE_EXTRACTORS}"

        # Build feature extractor
        self.feature_extractor = getattr(nets, feature_extractor_config.model)(
            **(vars(feature_extractor_config.kwargs) if feature_extractor_config.kwargs is not None else {})
        )

        # set `in_channels` for tasks to `out_channels` from feature extractor
        for task in vars(task_config).values():
            if not hasattr(task.module, "kwargs"):
                task.module.kwargs = SimpleNamespace()

            if not hasattr(task.module.kwargs, "in_channels"):
                task.module.kwargs.in_channels = feature_extractor_config.kwargs.out_channels

            # Module-specific setup
            if task.module.name == "SurfaceModule":
                task.module.kwargs.device = device

            elif task.module.name == "BiasFieldModule":
                raise NotImplementedError

                if "in_channels" not in task["module_kwargs"]:
                    pass
                if "target_shape" not in task["module_kwargs"]:
                    pass

        # Build tasks, e.g., segmentation, SR, surface
        self.tasks = torch.nn.ModuleDict(
            {
                name: getattr(tasks, task.module.name)(**vars(task.module.kwargs))
                for name, task in vars(task_config).items()
            }
        )

        biasfieldmodule = [task for task in self.tasks.values() if isinstance(task, tasks.BiasFieldModule)]
        # Register a forward hook for the desired activation.
        if (n := len(biasfieldmodule)) > 0:
            assert n == 1, "only one task of BiasFieldModule supported."
            biasfieldmodule = biasfieldmodule[0]
            if biasfieldmodule.feature_level != -1:
                self._feature_activations = {}

                def _activation_from_layer(name):
                    def hook(model, x, y):
                        self._feature_activations[name] = y
                    return hook

                # string, e.g., model.1.submodule.1.submodule.1
                self._feature_hook = self.feature_extractor.get_submodule(
                    task_config.feature_submodule
                ).register_forward_hook(_activation_from_layer("bias_field_features"))

    def forward(
        self,
        image: torch.Tensor,
        initial_vertices: torch.Tensor | None = None,
        hemispheres: tuple | None = None,
        # task_kwargs,
    ) -> dict:
        """

        Parameters
        ----------
        image : torch.Tensor
            The image to process.
        hemispheres :
            Hemispheres to predict with SurfaceModule.

        Returns
        -------
        pred : dict
            Dictionary containing the output of each task using the task name
            as the key.
        """
        features = self.feature_extractor(image)
        pred = {}
        for name, task in self.tasks.items():
            if isinstance(task, tasks.TaskModule):
                y = task(features)
            elif isinstance(task, tasks.BiasFieldModule):
                if task.feature_level == -1:
                    y = task(features)
                else:
                    intermediate_features = self._feature_activations[
                        "bias_field_features"
                    ]
                    y = task(intermediate_features)
                    self._feature_hook.remove()
            elif isinstance(task, tasks.ContrastiveModule):
                y = task(features)
            elif isinstance(task, tasks.SurfaceModule):
                y = task(features, initial_vertices, hemispheres)
            else:
                raise ValueError(f"Unknown task class {task}")
            pred[name] = y
        return pred
