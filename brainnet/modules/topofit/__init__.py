from brainnet.modules.graph.modules import UNetTransform, LinearTransform, SurfaceModule


class TopoFit(SurfaceModule):
    def __init__(
        self,
        in_channels: dict[str, int],
        out_channels: int = 3,
        white_channels: dict | None = None,
        pial_channels: list | None = None,
        *args,
        **kwargs,
        # image_shape: torch.IntTensor | torch.LongTensor,
        # config: None | config.TopoFitModelParameters = None,
    ) -> None:
        super().__init__(*args, **kwargs)

        # encoder=[64, 96, 128], ubend=160, decoder=[128, 96, 64]
        white_channels = (
            dict(
                encoder=[64, 64, 64],
                ubend=64,
                decoder=[64, 64, 64],
            )
            if white_channels is None
            else white_channels
        )
        pial_channels = [32] if pial_channels is None else pial_channels

        UNetTransform_kwargs = dict(channels=white_channels)

        for topo in self.active_topologies:  # e.g., 1, 2, ..., 7
            self.white_deform[str(topo)] = UNetTransform(
                sum(in_channels[j] for j in self.white_feature_maps[topo]),
                out_channels,
                self.topologies[: topo + 1],
                **UNetTransform_kwargs,
            )

        self.pial_deform = LinearTransform(
            sum(in_channels[j] for j in self.pial_feature_maps),
            out_channels,
            pial_channels,
        )
