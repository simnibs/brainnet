import torch

import brainnet.modules.head

import brainsynth.transforms
from brainsynth.transforms.utilities import channel_last
from brainsynth.transforms.spatial import ScaleAndSquare

class BrainReg(torch.nn.Module):
    def __init__(
        self,
        body: torch.nn.Module,
        svf: list[torch.nn.Module],
        device: str | torch.device,
    ):
        super().__init__()
        self.device = torch.device(device)
        assert len(svf) == len(body.feature_scales)
        self.body = body  # image feature extractor, e.g., unet
        self.svf = svf  # translates features to a stationary velocity field (SVF).
        self.SAS: None | ScaleAndSquare = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        images = torch.cat((img_0, img_1), dim=1)

        Parameters
        ----------
        images : torch.Tensor
            Either (N,1,W,H,D) or (N,2,W,H,D). Subsequent images are
            registered, e.g., when C=1, register images[0] and images[1]. When
            C=2, register images[0,0] and images[0,1], etc.

        Returns
        -------
        SVF : torch.Tensor
            Batch of predicted SVF, i.e., (N/2*C,3,W,H,D), such that an SVF
            aligns the first image to the second (and vice versa for -SVF).

        """
        # Estimate SVF for subsequent images.
        size = images.size()
        assert size[1] in {1,2}
        spatial_size = size[-3:]
        # Reshape from N,1,W,H,D to N/2,2,W,H,D
        images = images.reshape(-1, 2, *spatial_size)
        # assert images.size()[1] == 2, f"Exactly two images are required (got {images.size()[1]})"
        features = self.body(images)
        # predict an SVF for each feature map and sum them
        svf = self.body.sum_features([svf(f) for f,svf in zip(features, self.svf)])
        return svf

    def integrate_svf(self, svf):
        spatial_size = svf.size()[-3:]
        if not isinstance(self.SAS, ScaleAndSquare) or (self.SAS.grid.size() != spatial_size):
            grid = brainsynth.transforms.Grid(spatial_size, self.device)()
            self.SAS = ScaleAndSquare(grid, spatial_size, device=self.device)

        deform_fwd = self.SAS(svf)
        deform_bwd = self.SAS(-svf)

        return deform_fwd, deform_bwd

    @staticmethod
    def apply_affine(affine, grid):
        return grid @ affine[:3,:3].T + affine[:3, 3]


    def deform_image(self, image, deform):
        assert (spatial_size := deform.size()[-3:]) == image.size()[-3:]

        deformed_moving_grid = self.SAS.grid + channel_last(deform)
        sampler = brainsynth.transforms.GridSample(deformed_moving_grid, spatial_size)

        return sampler(image)

    def deform_surface(
        self,
        surface: torch.Tensor,
        deform: torch.Tensor,
        # spatial_size: None | torch.Size | torch.Tensor = None,
        # moving_aff = None,
        # moving_to_target_aff = None,
        # target_aff = None,
    ):
        """_summary_

        Parameters
        ----------
        surface : torch.Tensor
            ([N, ], V, 3) or ([N, ], V, 1, 1, 3)
        deform : torch.Tensor
            ([N, ], W, H, D, 3)

        Returns
        -------
        _type_
            _description_
        """

        # affine transformations

        # affine = moving_aff @ moving_to_target_aff @ torch.linalg.inv(target_aff)
        # moving image grid (voxels) deformed to target image voxel space
        # deformed_moving_grid = self.apply_affine(affine, grid_vox) + deform_fwd

        spatial_size = deform.size()[-3:]
        sampler = brainsynth.transforms.GridSample(surface, spatial_size)
        surface = surface + sampler(deform).mT

        return surface


class BrainNet(torch.nn.Module):
    def __init__(
        self,
        body: torch.nn.Module,
        heads: dict[str, torch.nn.Module],
        device: str | torch.device,
    ):
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
        self.device = torch.device(device)

        self.body = body
        self.heads = torch.nn.ModuleDict(heads)

        # biasfieldmodule = [
        #     h
        #     for h in self.heads.values()
        #     if isinstance(h, brainnet.modules.head.BiasFieldModule)
        # ]
        # # Register a forward hook for the desired activation.
        # if (n := len(biasfieldmodule)) > 0:
        #     assert n == 1, "only one task head of class BiasFieldModule supported."
        #     biasfieldmodule = biasfieldmodule[0]
        #     if biasfieldmodule.feature_level != -1:
        #         self._feature_activations = {}

        #         def _activation_from_layer(name):
        #             def hook(model, x, y):
        #                 self._feature_activations[name] = y

        #             return hook

        #         # string, e.g., model.1.submodule.1.submodule.1
        #         self._feature_hook = self.body.get_submodule(
        #             heads.feature_submodule
        #         ).register_forward_hook(_activation_from_layer("bias_field_features"))

    def forward(
        self,
        image: torch.Tensor,
        initial_vertices: None | dict = None,
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
        features = self.body(image)
        return self.forward_heads(features, initial_vertices)

    def forward_heads(self, features, initial_vertices, head_kwargs: dict | None = None):
        assert len(features) == 1
        features = features[0]

        pred = {}
        head_kwargs = head_kwargs or {}
        for name, head in self.heads.items():
            kwargs = head_kwargs[name] if name in head_kwargs else {}
            if isinstance(head, brainnet.modules.head.HeadModule):
                y = head(features, **kwargs)
            # elif isinstance(head, brainnet.modules.head.BiasFieldModule):
            #     if head.feature_level == -1:
            #         y = head(features, **kwargs)
            #     else:
            #         intermediate_features = self._feature_activations[
            #             "bias_field_features"
            #         ]
            #         y = head(intermediate_features, **kwargs)
            #         self._feature_hook.remove()
            elif isinstance(head, brainnet.modules.head.ContrastiveModule):
                y = head(features, **kwargs)
            elif isinstance(head, brainnet.modules.head.surface_modules):
                y = head(features, initial_vertices, **kwargs)
            else:
                raise ValueError(f"Unknown head class {head}")
            pred[name] = y
        return pred
