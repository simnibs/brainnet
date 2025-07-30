import torch

from brainnet.mesh.surface import Surface
from brainnet import resources
import brainnet.modules.image
import brainnet.modules.graph


class TopoFit(torch.nn.Module):
    def __init__(
        self, unet_kwargs: dict, graph_kwargs: dict, group_output_by: str = "surface"
    ):
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

        self.set_group_output_by(group_output_by)

    def set_group_output_by(self, group_output_by: str):
        assert group_output_by in ("surface", "hemisphere")
        self._group_output_by = group_output_by
        self._swap_output = self._group_output_by == "surface"

    @staticmethod
    def swap_output_levels(out):
        """Swap to first two levels of a dictionary, e.g.,

            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}}

        to

            {"x": {"a": 1, "b": 3}, "y": {"a": 2, "b": 4}}

        Assumes that all subdicts has the same entries!
        """
        level0 = tuple(out.keys())
        level1 = tuple(out[level0[0]].keys())
        return {s: {h: out[h][s] for h in level0} for s in level1}

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
        out = self.graph(self.unet(image), template)
        return self.swap_output_levels(out) if self._swap_output else out

    @classmethod
    def from_pretrained(
        cls, contrast: str, resolution: str, device: str | torch.device = "cpu"
    ):
        device = torch.device(device)
        state = resources.load_pretrained_state("topofit", contrast, resolution, device)
        config = resources.load_pretrained_config("topofit", contrast, resolution)

        model = cls(config["model"]["unet"], config["model"]["topofit"])
        model.to(device)
        model.load_state_dict(state)

        return model
