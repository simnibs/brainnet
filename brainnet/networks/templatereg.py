import torch
from brainsynth.utilities import apply_affine
from brainnet.modules.image.unet import UNet
from brainnet import resources

# Transformation that converts lh to rh in template space (MNI305).
# Estimated by least squares registration of lh.white to rh.white from the
# templates in brainsynth/resources/template/surf
template_lh_to_rh = torch.tensor(
    [
        [
            [-1.0338, 0.0032, -0.0110, -0.0359],
            [-0.0462, 0.9972, 0.0115, -0.4072],
            [0.0088, -0.0099, 1.0198, 0.6601],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    ]
)


class TemplateRegAffine(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        pre_conv_ch: int = 8,
        encoder_channels: list[list[int]] | tuple = ((32,), (32,), (32,), (32,), (32,)),
        decoder_channels: list[list[int]] | tuple = ((32,), (32,), (32,), (32,)),
        points_per_hemisphere: int = 12,
        hemispheres: str | tuple[str, str] = ("lh", "rh"),
        weigh_by_feature_mass: bool = True,
    ):
        """Reimplementation of Andrew Hoopes' original template alignment tool
        from the DeepSurfer package.

        Given an MR image and its affine, this network estimates an affine
        world-to-world transformation from MNI305 to subject space.

        """
        super().__init__()
        _valid_hemispheres = {"lh", "rh"}
        if isinstance(hemispheres, str):
            self.hemispheres = (hemispheres,)
        else:
            self.hemispheres = hemispheres
        self.n_hemi = len(hemispheres)
        assert all(h in _valid_hemispheres for h in self.hemispheres)
        assert self.n_hemi <= 2
        self.weigh_by_feature_mass = weigh_by_feature_mass
        self.pph = points_per_hemisphere
        self.ppb = self.pph * self.n_hemi
        self.spatial_dims = (-3, -2, -1)

        # Image feature extraction. Each feature is learned so as to "zoom in"
        # on a salient region of a subject-specific image which is useful for
        # predicting the registration.

        # Pre-convolve image with strided conv (downsampling)
        self.image_fx_pre = torch.nn.Sequential(
            torch.nn.Conv3d(1, pre_conv_ch, kernel_size=3, stride=2, padding=1),
            torch.nn.PReLU(),
        )
        self.image_fx_unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=pre_conv_ch,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )

        # Post-convolve to the number of predicted points
        self.image_fx_post = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.image_fx_unet.final_channels, self.ppb, kernel_size=3, padding=1
            ),
            # NOTE
            # ReLU is important as to ensure that the feature maps contain no
            # negative values and hence are valid as weights in a weighted
            # average
            torch.nn.ReLU(),
        )

        # Template points. Learns the coordinates (in template world space)
        # corresponding to the salient regions extracted from the image of a
        # particular subject

        # Approximate bounding box of the template surfaces
        bbox = dict(
            lh=torch.distributions.Uniform(
                low=torch.tensor([-80.0, -120.0, -60.0]),
                high=torch.tensor([0.0, 80.0, 90.0]),
            ),
            rh=torch.distributions.Uniform(
                low=torch.tensor([0.0, -120.0, -60.0]),
                high=torch.tensor([80.0, 80.0, 90.0]),
            ),
        )
        self.template_points = torch.nn.ParameterDict(
            {h: torch.nn.Parameter(bbox[h].sample([self.pph])) for h in hemispheres}
        )

        # Image grid for computing feature barycenters in voxel space
        self.image_grid = torch.nn.Buffer(torch.tensor([]), persistent=False)
        self.image_shape = torch.Size([])
        # For estimating the affine
        self.ones = torch.nn.Buffer(torch.ones((1, self.ppb, 1)), persistent=False)

    def split_hemispheres(self, t: torch.Tensor, dim: int = 1):
        return dict(zip(self.hemispheres, t.split(self.pph, dim)))

    def _estimate_targets(self, image, vox2ras):
        # Image features which "zoom in" on characteristic parts of the image
        features = self.image_fx_pre(image)
        features = self.image_fx_unet(features)["dec:3"]
        features = self.image_fx_post(features)

        # precompute and cache mesh grid for computing barycenters. only need to update
        # this when the input image shape changes
        if self.image_shape != (spatial_shape := features.shape[-3:]):
            self.image_grid = torch.stack(
                torch.meshgrid(
                    # The predicted feature maps are half the resolution of the
                    # input image
                    [
                        torch.arange(0, 2 * s, 2, device=image.device)
                        for s in spatial_shape
                    ],
                    indexing="ij",
                )
            )[None]  # add a batch dim
            self.image_shape = spatial_shape

        # Weighted average of features
        feature_mass = features.sum(self.spatial_dims, keepdim=True).abs() + 1e-6
        features = features / feature_mass
        feature_mass = feature_mass.squeeze(self.spatial_dims)
        feature_mass = feature_mass[..., None] / feature_mass.sum()
        # print("mass", feature_mass)
        # print("norm", features.sum((2,3,4)))
        # (batch, n_channels, 3)
        barycenters = torch.sum(
            self.image_grid[:, None] * features[:, :, None], self.spatial_dims
        )
        if self.weigh_by_feature_mass:
            self._wls_weight = self.split_hemispheres(feature_mass)
        else:
            self._wls_weight = None
        # print(barycenters.amin(1), barycenters.amax(1))

        # subject specific barycenters (target points) in RAS
        return self.split_hemispheres(apply_affine(vox2ras, barycenters))

    def forward(self, image, vox2ras):
        targets = self._estimate_targets(image, vox2ras)

        affines = self.estimate_affine_hemispheres(targets)
        if self.n_hemi == 2:
            affines["brain"] = self.estimate_affine_brain(targets)
        return affines

    def estimate_affine(
        self,
        template: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ):
        """Estimate the affine transformation that aligns the template points
        with the target points, optionally weighted by `weight`.


        """
        n_batch = target.shape[0]
        template = template.unsqueeze(0).expand((n_batch, *template.shape))

        A = torch.cat((template, self.ones[:, : template.shape[1]]), -1)
        B = torch.cat((target, self.ones[:, : target.shape[1]]), -1)

        if weight is not None:
            # Solve the weighted least squares problem instead
            # https://en.wikipedia.org/wiki/Weighted_least_squares
            A = A * weight
            B = B * weight

        result = torch.linalg.lstsq(A, B)
        return result.solution.mT

    def estimate_affine_hemispheres(self, target: dict[str, torch.Tensor]):
        """Given the (learned) template points, fit an affine transform to the
        predicted target points.

        Parameters
        ----------
        target :
            Dictionary with lh and rh containing tensor of shape
            (B, points_per_hemisphere, 3).

        Returns
        -------
        """
        return {
            h: self.estimate_affine(
                self.template_points[h], target[h], self._wls_weight[h]
            )
            for h in self.template_points
        }

    def estimate_affine_brain(self, target: dict[str, torch.Tensor]):
        return self.estimate_affine(
            torch.cat(tuple(self.template_points.values()), dim=0),
            torch.cat(tuple(target.values()), dim=1),
            torch.cat(tuple(self._wls_weight.values()), dim=1),
        )

    @classmethod
    def from_pretrained(
        cls,
        contrast: str = "synth",
        resolution: str = "random",
        suffix: str | None = None,
        device: str | torch.device = "cpu",
    ):
        device = torch.device(device)
        state = resources.load_pretrained_state(
            "trega", contrast, resolution, suffix, device
        )
        config = resources.load_pretrained_config("trega", contrast, resolution, suffix)

        model = cls(**config["model"])
        model.to(device)
        model.load_state_dict(state)

        return model
