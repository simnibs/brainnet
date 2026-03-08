import torch

from brainsynth.transforms.spatial import FlipImage, FlipSurface
from brainsynth.transforms.utils import recursive_function
from brainnet.mesh.surface import Surface
from brainnet import resources
import brainnet.modules.image
import brainnet.modules.graph

recursive_float = recursive_function(torch.Tensor.float)


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

            {"x": {"a": 1, "b": 3}, "y": {"`a": 2, "b": 4}}

        Assumes that all subdicts has the same entries!
        """
        level0 = tuple(out.keys())
        level1 = tuple(out[level0[0]].keys())
        return {s: {h: out[h][s] for h in level0} for s in level1}

    def forward(
        self,
        image: torch.Tensor,
        template: dict[str, torch.Tensor],
        return_pial: bool = True,
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
        features = self.unet(image)
        with torch.autocast("cuda", enabled=False):
            features = recursive_float(features)
            out = self.graph(features, template, return_pial)
        return self.swap_output_levels(out) if self._swap_output else out

    @classmethod
    def from_pretrained(
        cls,
        contrast: str,
        resolution: str,
        suffix: str | None = None,
        device: str | torch.device = "cpu",
    ):
        device = torch.device(device)
        state = resources.load_pretrained_state(
            "topofit", contrast, resolution, suffix, device
        )
        config = resources.load_pretrained_config(
            "topofit", contrast, resolution, suffix
        )

        model = cls(config["model"]["unet"], config["model"]["topofit"])
        model.to(device)
        model.load_state_dict(state)

        return model


class TopoFitSingleHemi(TopoFit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """TopoFit on a single hemisphere only. The network is designed to
        *always* predict on left hemisphere, hence, to predict on the right
        hemisphere, the image will first be flipped, and the surface flipped
        back.
        """
        # [BATCH,CHANNELS,]RAS so LR = -3
        self._lr_dim = -3
        self._flip_image = FlipImage(self._lr_dim)
        self._flip_vertices = FlipSurface(self._lr_dim)
        # flip the spherical registration initialization
        # (Only the vertices are used)
        self.graph.sphere_reg["rh"] = self.graph.sphere_reg["rh"].flip(self._lr_dim)
        # set rh topology equal to lh! (we flip the output after the forward
        # pass of `super`)
        self.graph.out_topology["rh"] = self.graph.out_topology["lh"]

    def _flip_surface(self, surface, image_size=None):
        if isinstance(surface, Surface):
            surface = surface.flip(self._lr_dim, image_size)
        else:
            surface = self._flip_vertices(surface, image_size)
        return surface

    def _flip_input(self, image, template, size):
        """Force lh"""
        image = self._flip_image(image)
        template["rh"] = self._flip_surface(template["rh"], size)
        return image, template

    def _flip_output(self, out, size):
        """lh to rh."""

        # This also flips vector data associated with each surface
        for s in {"white", "pial"}:
            out[s]["rh"] = self._flip_surface(out[s]["rh"], size)
        # out["pial"]["rh"] = self._flip_surface(out["pial"].pop("lh"), size)
        # no volume geometry
        out["sphere.reg"]["rh"] = self._flip_surface(out["sphere.reg"]["rh"])
        return out

    def forward(
        self, image: torch.Tensor, template: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, Surface]]:
        """Estimate cortical surfaces from `image`.

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
        assert len(template) == 1  # one hemi at a time
        if is_rh := ("rh" in template):
            size = image.shape[-3:]
            image, template = self._flip_input(image, template, size)

        out = super().forward(image, template)

        return self._flip_output(out, size) if is_rh else out
