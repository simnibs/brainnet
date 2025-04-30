from brainnet.modules.graph.modules import UNetTransform, SurfaceModule
import brainnet.modules.graph.layers


# class TopoInit(SurfaceInitializerModule):
#     def __init__(
#         self,
#         in_channels: dict[str, int],
#         out_channels: int = 3,
#         channels: dict | None = None,
#         *args,
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)

#         UNetTransform_kwargs = dict(channels=channels)

#         for topo in self.all_topologies:  # e.g., 1, 2, ..., 7
#             self.deform[str(topo)] = UNetTransform(
#                 sum(in_channels[j] for j in self.feature_maps[topo]),
#                 out_channels,
#                 self.topologies[: topo + 1],
#                 **UNetTransform_kwargs,
#             )


class TopoFit(SurfaceModule):
    def __init__(
        self,
        in_channels: dict[str, int],
        # out_channels: int = 3,
        out_channels: int = 6,
        white_channels: dict | None = None,
        pial_channels: list | None = None,
        pial_deform_module: str = "LinearDeformationBlock",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if white_channels is None:
            white_channels = dict(encoder=[64, 64, 64, 64], decoder=[64, 64, 64])
        pial_channels = [32] if pial_channels is None else pial_channels

        UNetTransform_kwargs = dict(channels=white_channels)
        UNetTransform_kwargs = dict(
            channels=white_channels,
            deform_init_values=3 * [0.0] + 3 * [0.01],
        )

        for topo in self.all_topologies:  # e.g., 1, 2, ..., 7
            self.white_deform[str(topo)] = UNetTransform(
                sum(in_channels[j] for j in self.white_feature_maps[topo]),
                out_channels,
                self.topologies[: topo + 1],
                **UNetTransform_kwargs,
            )

        pial_init_values = 3 * [0.01] + 3 * [0.01]
        m = getattr(brainnet.modules.graph.layers, pial_deform_module)
        match pial_deform_module:
            case "LinearDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    out_channels,
                    out_init_values=pial_init_values,
                )
            case "GraphConvolutionDeformationBlock" | "EdgeConvolutionDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    out_channels,
                    self.out_topology.conv_index_reduce,
                    self.out_topology.conv_index_gather,
                    out_init_values=pial_init_values,
                )
            case "ResidualGraphConvolutionDeformationBlock":
                self.pial_deform = m(
                    sum(in_channels[j] for j in self.pial_feature_maps),
                    pial_channels,
                    out_channels,
                    self.out_topology.conv_index_reduce,
                    self.out_topology.conv_index_gather,
                    n_residual_blocks=3,
                    out_init_values=pial_init_values,
                )
            case _:
                raise ValueError(
                    f"Invalid module for pial deformation ({pial_deform_module})"
                )
