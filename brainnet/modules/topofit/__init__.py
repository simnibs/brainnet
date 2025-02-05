from brainnet.modules.graph.modules import UNetTransform, SurfaceModule
import brainnet.modules.graph.layers

class TopoFit(SurfaceModule):
    def __init__(
        self,
        in_channels: dict[str, int],
        out_channels: int = 3,
        white_channels: dict | None = None,
        pial_channels: list | None = None,
        pial_deform_module: str = "LinearDeformationBlock",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        white_channels = (
            dict(
                encoder=[96, 96, 96, 96],
                decoder=[96, 96, 96],
            )
            if white_channels is None
            else white_channels
        )
        pial_channels = [32] if pial_channels is None else pial_channels

        UNetTransform_kwargs = dict(channels=white_channels)

        for topo in self.all_topologies:  # e.g., 1, 2, ..., 7
            self.white_deform[str(topo)] = UNetTransform(
                sum(in_channels[j] for j in self.white_feature_maps[topo]),
                out_channels,
                self.topologies[: topo + 1],
                **UNetTransform_kwargs,
            )

        m = getattr(brainnet.modules.graph.layers, pial_deform_module)
        match pial_deform_module:
            case "LinearDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    out_channels,
                )
            case "GraphConvolutionDeformationBlock" | "EdgeConvolutionDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    out_channels,
                    self.out_topology.conv_index_reduce,
                    self.out_topology.conv_index_gather,
                )
            case "ResidualGraphConvolutionDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    out_channels,
                    self.out_topology.conv_index_reduce,
                    self.out_topology.conv_index_gather,
                    n_residual_blocks=3,
                )
            case _:
                raise ValueError(f"Invalid module for pial deformation ({pial_deform_module})")