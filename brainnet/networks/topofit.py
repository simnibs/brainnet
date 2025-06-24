import torch

from brainnet.mesh.surface import Surface
from brainnet import resources
import brainnet.modules.image
import brainnet.modules.graph


class TopoFit(torch.nn.Module):
    def __init__(self, unet_kwargs: dict, graph_kwargs: dict):
        """

        Parameters
        ----------
        unet_kwargs: dict
            Kwargs pased to brainnet.modules.body.UNet.
        graph_kwargs: dict
            Kwargs passed to brainnet.modules.graph.TopoFit.

        """
        super().__init__()
        self.unet = brainnet.modules.image.UNet(**unet_kwargs)

        graph_kwargs["in_channels"] = self.unet.num_features
        graph_kwargs["white_feature_maps"] = [self.unet.decoder_features] * (
            graph_kwargs["max_order"] - graph_kwargs["in_order"] + 1
        )
        graph_kwargs["pial_feature_maps"] = self.unet.decoder_features

        self.graph = brainnet.modules.graph.TopoFit(**graph_kwargs)

    def forward(
        self, image: torch.Tensor, template: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, Surface]]:
        """Estimate cortical surfaces on `image`.

        Parameters
        ----------
        image : torch.Tensor
            The image on which to estimate the surfaces.
        template : dict[str, torch.Tensor]
            Template positions for vertices coregistered to `image`.

        Returns
        -------
        surfaces : dict[str, dict[str, Surface]]
            Return a dictionary with the following signature

                dict[surface type, dict[hemisphere, Surface]]
        """
        return self.graph(self.unet(image), template)

    @classmethod
    def from_pretrained_model(
        cls, contrast: str, resolution: str, device: str | torch.device = "cpu"
    ):
        device = torch.device(device)
        state = resources.load_pretrained_state("topofit", contrast, resolution, device)
        config = resources.load_pretrained_config("topofit", contrast, resolution)

        model = cls(config["model"]["unet_kwargs"], config["model"]["topofit_kwargs"])
        model.to(device)
        model.load_state_dict(state)

        return model
