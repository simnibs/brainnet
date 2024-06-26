from types import SimpleNamespace

import torch

from brainnet.modules import head

VALID_BASE_NETS = {
    "UNet",
    "ResNet",
}

from brainnet.modules.body.unet import UNet

class BrainNet(torch.nn.Module):
    def __init__(self, body_config, head_config, device):
        """Construct the BrainNet model consisting of body (image feature
        extractor) and one or more heads (task specific predictors).

        Parameters
        ----------
        body_config : _type_
            _description_
        task_config : _type_
            _description_
        """
        super().__init__()

        device = torch.device(device)

        assert (
            body_config.model in VALID_BASE_NETS
        ), f"Invalid feature network. Please chose from {VALID_BASE_NETS}"

        # Build feature extractor
        # self.body = getattr(nets, body_config.model)(
        #     **(vars(body_config.kwargs) if body_config.kwargs is not None else {})
        # )
        self.body = UNet(
            **(vars(body_config.kwargs) if body_config.kwargs is not None else {})
        )

        # set `in_channels` for heads to `out_channels` from feature extractor
        for h in vars(head_config).values():
            if not hasattr(h.module, "kwargs"):
                h.module.kwargs = SimpleNamespace()

            if not hasattr(h.module.kwargs, "in_channels"):
                # task.module.kwargs.in_channels = body_config.kwargs.out_channels
                h.module.kwargs.in_channels = body_config.kwargs.decoder_channels[-1][-1]

            # Module-specific setup
            if h.module.name in {"SurfaceModule", "SurfaceAdjustModule"}:
                h.module.kwargs.device = device

            elif h.module.name == "BiasFieldModule":
                raise NotImplementedError

                if "in_channels" not in h["module_kwargs"]:
                    pass
                if "target_shape" not in h["module_kwargs"]:
                    pass

        # Build heads, e.g., segmentation, SR, surface
        self.heads = torch.nn.ModuleDict(
            {
                name: getattr(head, head.module.name)(**vars(head.module.kwargs))
                for name, head in vars(head_config).items()
            }
        )

        biasfieldmodule = [head for head in self.heads.values() if isinstance(head, head.BiasFieldModule)]
        # Register a forward hook for the desired activation.
        if (n := len(biasfieldmodule)) > 0:
            assert n == 1, "only one task head of class BiasFieldModule supported."
            biasfieldmodule = biasfieldmodule[0]
            if biasfieldmodule.feature_level != -1:
                self._feature_activations = {}

                def _activation_from_layer(name):
                    def hook(model, x, y):
                        self._feature_activations[name] = y
                    return hook

                # string, e.g., model.1.submodule.1.submodule.1
                self._feature_hook = self.body.get_submodule(
                    head_config.feature_submodule
                ).register_forward_hook(_activation_from_layer("bias_field_features"))

    def forward(
        self,
        image: torch.Tensor,
        initial_vertices: None | dict = None,
        head_kwargs = None,
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
        pred = {}
        head_kwargs = head_kwargs or {}

        features = self.body(image)

        for name, head in self.heads.items():
            kwargs = head_kwargs[name] if name in head_kwargs else {}
            if isinstance(head, head.HeadModule):
                y = head(features, **kwargs)
            elif isinstance(head, head.BiasFieldModule):
                if head.feature_level == -1:
                    y = head(features, **kwargs)
                else:
                    intermediate_features = self._feature_activations[
                        "bias_field_features"
                    ]
                    y = head(intermediate_features, **kwargs)
                    self._feature_hook.remove()
            elif isinstance(head, head.ContrastiveModule):
                y = head(features, **kwargs)
            elif isinstance(head, head.surface_modules):
                y = head(features, initial_vertices, **kwargs)
            else:
                raise ValueError(f"Unknown head class {head}")
            pred[name] = y
        return pred
