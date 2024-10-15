import torch

from brainnet.modules.graph import UNetTransform, LinearTransform, SurfaceModule

class TopoFit(SurfaceModule):
    def __init__(
        self,
        in_channels: dict[str, int],
        out_res: int = 6,
        white_kwargs: dict | None = None,
        pial_kwargs: dict | None = None,
        device: str | torch.device = "cpu",
        # image_shape: torch.IntTensor | torch.LongTensor,
        # config: None | config.TopoFitModelParameters = None,
    ) -> None:
        super().__init__(out_res, white_kwargs, pial_kwargs, device)

        out_channels = 3

        white_channels = dict(
            # encoder=[64, 96, 128], ubend=160, decoder=[128, 96, 64]
            encoder=[64, 64, 64],
            ubend=64,
            decoder=[64, 64, 64],
        )
        pial_channels = [32]

        UNetTransform_kwargs = dict(channels=white_channels)

        self.white_deform = torch.nn.ModuleList()
        for i in range(self.n_topologies):
            self.white_deform.append(
                UNetTransform(
                    sum(in_channels[j] for j in self.white_feature_maps[i]),
                    out_channels,
                    self.topologies[: i + 1],
                    **UNetTransform_kwargs,
                )
            )

        self.pial_deform = LinearTransform(
            sum(in_channels[j] for j in self.pial_feature_maps),
            out_channels,
            pial_channels,
        )
