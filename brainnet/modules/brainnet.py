import copy

import nibabel as nib
import torch


import brainsynth
import brainsynth.transforms
from brainsynth.transforms.utilities import channel_last
from brainsynth.transforms.spatial import ScaleAndSquare

import brainnet.modules.head
import brainnet.mesh.topology
from brainnet.mesh.surface import TemplateSurfaces
from brainnet.modules.graph.modules import UNetTransform
from brainnet.sphere_utils import change_sphere_size


class BrainReg(torch.nn.Module):
    def __init__(
        self,
        body: torch.nn.Module,
        svf: list[torch.nn.Module],
        device: str | torch.device,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.body = body  # image feature extractor, e.g., unet
        self.svf = svf  # translates features to a stationary velocity field (SVF).
        self.svf_scales = body.decoder_scale[-len(svf) :]
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
        assert size[1] in {1, 2}
        spatial_size = size[-3:]
        # Reshape from N,1,W,H,D to N/2,2,W,H,D
        images = images.reshape(-1, 2, *spatial_size)
        # assert images.size()[1] == 2, f"Exactly two images are required (got {images.size()[1]})"
        features = self.body(images)
        # predict an SVF for each feature map and sum them
        svf = torch.zeros((images.shape[0], 3, *spatial_size), device=images.device)
        for svf_module, scale in zip(self.svf, self.svf_scales):
            svf = svf + self.body.upsample_feature(svf_module(features), scale)
        return svf

    def integrate_svf(self, svf):
        spatial_size = svf.size()[-3:]
        if not isinstance(self.SAS, ScaleAndSquare) or (
            self.SAS.grid.size() != spatial_size
        ):
            grid = brainsynth.transforms.Grid(spatial_size, self.device)()
            self.SAS = ScaleAndSquare(grid, spatial_size, n_steps=4, device=self.device)

        deform_fwd = self.SAS(svf)
        deform_bwd = self.SAS(-svf)

        return deform_fwd, deform_bwd

    @staticmethod
    def apply_affine(affine, grid):
        return grid @ affine[:3, :3].T + affine[:3, 3]

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

    def forward_heads(
        self, features, initial_vertices, head_kwargs: dict | None = None
    ):
        # assert len(features) == 1
        # features = features[0]

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


class BrainInflate(torch.nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        n_steps: int = 10,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        n_topologies = 7
        in_channels = 9  # coordinates, normals, LBO

        # assert n_topologies >= out_res + 1
        # self.out_res = out_res

        # The topology is defined on the left hemisphere and although the
        # topology is the same for both hemispheres, we need to reverse the
        # order of the vertices in face array in order for the ordering to
        # remain consistent (e.g., counter-clockwise) once the vertices are
        # (almost) left-right mirrored

        # We use the left topology in the submodules which only use knowledge
        # of the neighborhoods to define the convolutions (and this is
        # independent of the face orientation).
        self.topologies = brainnet.mesh.topology.get_recursively_subdivided_topology(
            n_topologies - 1,
            brainnet.mesh.topology.initial_faces.to(self.device),
        )
        self.topologies = self.topologies[:-1]

        top = self.topologies[-1]
        self.topology = dict(lh=top, rh=copy.deepcopy(top))
        self.topology["rh"].reverse_face_orientation()

        self.surface = {
            k: TemplateSurfaces(torch.zeros(v.n_vertices, 3, device=self.device), v)
            for k, v in self.topology.items()
        }

        self.n_steps = n_steps
        self.step_size = 1.0 / n_steps

        channels = dict(
            encoder=[16, 32, 64],
            ubend=128,
            decoder=[64, 32, 16],
        )
        UNetTransform_kwargs = dict(channels=channels, n_convolutions=2)

        self.transform = UNetTransform(
            in_channels, out_channels, self.topologies, **UNetTransform_kwargs
        )

    def set_surfaces(self, vertices):
        self.surface = {
            k: TemplateSurfaces(v, self.topology[k]) for k, v in vertices.items()
        }

    def normalize_coordinates(self):
        """Normalize coordinates by centering the surface on the origin and
        dividing by the maximum along in each dimension.
        """
        self.bbox = {k: v.bounding_box() for k, v in self.surface.items()}
        self.center = {k: v.mean(1)[:, None] for k, v in self.bbox.items()}
        self.size = {k: self.bbox[k][:, [1]] - self.center[k] for k in self.bbox}
        for h in self.surface:
            self.surface[h].vertices -= self.center[h]
            self.surface[h].vertices /= self.size[h]

    def unnormalize_coordinates(self, undo_scale=True, undo_center=False):
        for h in self.surface:
            if undo_scale:
                self.surface[h].vertices *= self.size[h]
            if undo_center:
                self.surface[h].vertices += self.center[h]

    def compute_features(self, surface):
        normals = surface.compute_vertex_normals()
        lbo = surface.compute_laplace_beltrami_operator()
        return torch.concat((surface.vertices.mT, normals.mT, lbo.mT), dim=1)

    @staticmethod
    def update_coordinates(v, dv):
        return v + dv.mT

    def _forward_hemi(self, hemi):
        surface = self.surface[hemi]
        surface.vertex_data["sulc"] = torch.zeros(
            (surface.n_batch, surface.topology.n_vertices), device=self.device
        )
        for _ in range(self.n_steps):
            features = self.compute_features(surface)
            dV = self.transform(features)
            surface.vertices = self.update_coordinates(
                surface.vertices, self.step_size * dV
            )
            # average convexity
            # we already computed the normals as features (3-6)!
            surface.vertex_data["sulc"] += self.step_size * torch.sum(
                features[:, 3:6] * dV, 1
            )

    def forward(self, vertices: dict[str, torch.Tensor]):
        self.set_surfaces(vertices)
        self.normalize_coordinates()

        for h in self.surface:
            self._forward_hemi(h)

        self.unnormalize_coordinates()

        return self.surface


class SphericalReg(torch.nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        n_topologies = 6
        in_channels = 3  # xyz coordinates
        n_steps = 1

        self.n_steps = n_steps
        self.step_size = 1.0 / n_steps

        # assert n_topologies >= out_res + 1
        # self.out_res = out_res
        self.set_topologies(n_topologies)

        self.surface = {
            k: TemplateSurfaces(torch.zeros(v.n_vertices, 3, device=self.device), v)
            for k, v in self.topology.items()
        }

        # Initial positions
        self.sphere = TemplateSurfaces(
            torch.tensor(
                nib.freesurfer.read_geometry(
                    brainsynth.resources_dir / "sphere-reg.srf"
                )[0][: self.topology["lh"].n_vertices],
                dtype=torch.float,
                device=self.device,
            ),
            self.topology["lh"],
        )
        self.prepare_sphere()

        self.moved = copy.deepcopy(self.sphere)

        # self.nn_tris = torch.load(
        #     brainnet.resources_dir / f"template.{n_topologies - 1}.nn_tri.pt"
        # ).to(self.device)


        # channels = dict(
        #     encoder=[128, 128, 128],
        #     ubend=128,
        #     decoder=[128, 128, 128],
        # )
        channels = dict(
            encoder=[32, 64, 128],
            ubend=256,
            decoder=[128, 64, 32],
        )
        UNetTransform_kwargs = dict(channels=channels, n_convolutions=2)

        self.transform = UNetTransform(
            in_channels, out_channels, self.topologies, **UNetTransform_kwargs
        )

    def set_topologies(self, n_topologies):
        # The topology is defined on the left hemisphere and although the
        # topology is the same for both hemispheres, we need to reverse the
        # order of the vertices in face array in order for the ordering to
        # remain consistent (e.g., counter-clockwise) once the vertices are
        # (almost) left-right mirrored

        # We use the left topology in the submodules which only use knowledge
        # of the neighborhoods to define the convolutions (and this is
        # independent of the face orientation).
        self.topologies = brainnet.mesh.topology.get_recursively_subdivided_topology(
            n_topologies - 1,
            brainnet.mesh.topology.initial_faces.to(self.device),
        )

        top = self.topologies[-1]
        self.topology = dict(lh=top, rh=copy.deepcopy(top))
        self.topology["rh"].reverse_face_orientation()

    def set_surfaces(self, vertices):
        for k,v in vertices.items():
            self.surface[k].vertices = v
            self.surface[k].normalize_to_bounding_box()

    def prepare_sphere(self):
        # map to unit sphere
        self.sphere.center_on_origin()
        change_sphere_size(self.sphere, radius=1.0)

        # indices, indptr, counts = self.sphere.topology.get_vertexwise_faces()
        # self.sphere_indices = indices
        # self.sphere_indptr = indptr
        # self.sphere_tricounts = counts

        # self.spherical_coords = cart_to_sph(self.sphere.vertices)

    # def normalize_from_quantiles(self, t, low=0.01, high=0.99):
    #     ql = t.quantile(low)
    #     qu = t.quantile(high)
    #     return torch.clip((t - ql) / (qu - ql), 0.0, 1.0)

    def compute_features(self, feature_surface, interpolate_features: bool = True):
        # K = surface.compute_laplace_beltrami_operator()
        # H = surface.compute_mean_curvature(K)
        # H = surface.compute_iterative_spatial_smoothing(H, iterations=5)
        # H = self.normalize_from_quantiles(H)
        # return torch.concat((surface.vertices.mT, H[:, None]), dim=1)

        # if interpolate_features:
        #     # Interpolate features at the location of the
        #     if self.device == "cpu":

        #         tree = cKDTree(self.moved.vertices.numpy())
        #         _, nn = tree.query(self.sphere.vertices.numpy())
        #         nn = torch.tensor(nn)
        #     else:
        #         # for each v in vertices find the closest point in sphere.vertices
        #         nn = self.sphere.nearest_neighbor(self.moved)

        #     proj_tri, proj_w = self.moved.project_points(
        #         self.sphere.vertices,
        #         self.nn_tris[nn],
        #         return_proj=False,
        #         return_dist=False,
        #     )

        #     tris = feature_surface.vertices[
        #         feature_surface.batch_ix, feature_surface.faces[proj_tri]
        #     ]
        #     features = torch.sum(tris * proj_w[..., None], dim=-2)
        # else:
        #     # map features
        #     features = feature_surface.vertices

        features = feature_surface.vertices

        return features.mT

    def update_coordinates(self, dv):
        # vertices : (batch, vertices, coordinates)
        # dv : (batch, channels, vertices)
        dv = dv.mT
        # k = dv[..., :3]
        # alpha = dv[..., [3]]
        # print(k[0, :10])
        # print(alpha[0, :10])
        # self.moved.rotate(k, alpha, inplace=True)
        for i in range(3):
            self.moved.rotate_dim(i, dv[..., [i]], inplace=True)

    def _forward_hemi(self, hemi):
        # initialize positions
        self.moved.vertices = self.sphere.vertices.clone()
        # features, surf in original space
        feature_surface = self.surface[hemi]

        for i in range(self.n_steps):
            # interpolate features
            # (do not interpolate in first iteration!)
            features = self.compute_features(feature_surface, i > 0)
            dV = self.transform(features)
            self.update_coordinates(self.step_size * dV)
        return self.moved.vertices

    def forward(self, vertices: dict[str, torch.Tensor]):
        self.set_surfaces(vertices)

        return {h: self._forward_hemi(h) for h in vertices}


# from torch.distributions.uniform import Uniform

# class BrainAlign(torch.nn.Module):
#     def __init__(self, device: str | torch.device, points_per_hemisphere: int = 12):

#         self.device = torch.device(device)

#         # MNI152 template
#         self.hemispheres = ("lh", "rh")

#         # BOUNDING BOX within which to sample initial points. Uniformly? 12 points perh hemi
#         # L: [-80.0, 0.0], [-120.0, 80.0], [-60.0, 90.0]
#         # R: [0.0, 80.0], [-120.0, 80.0], [-60.0, 90.0]

#         bbox_brain = dict(
#             lh=torch.tensor([[-80.0, 0.0], [-120.0, 80.0], [-60.0, 90.0]], device=self.device),
#             rh=torch.tensor([[0.0, 80.0], [-120.0, 80.0], [-60.0, 90.0]], device=self.device),
#         )
#         self.template_points = {
#             h: torch.nn.Parameter(
#                 torch.stack(
#                     [Uniform(low, high, shape=(points_per_hemisphere,)) for low,high in bbox_brain[h]]
#                 ),
#                 requires_grad=True
#             )  for h in hemi}
#         self.predicted_points = {h: torch.zeros_like(p) for h,p in self.template_points.items()}
#         self.coordinates = None
#         self.image_shape = torch.Size([])


#         self.unet =
#         self.

#     def WLS(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor):
#         """Solve a weighted least squares problem.

#             Y = XB

#         Parameters
#         ----------
#         X : torch.Tensor
#             Predicted points.
#         Y : torch.Tensor
#             The learned template points.
#         W : torch.Tensor
#             Weighted (salience) of each predicted point.
#         """
#         XTW = X.T * (W / W.sum())
#         return torch.linalg.inv(XTW @ X) @ XTW @ Y

#     def forward(self, image, vox_to_mri, hemi=None):
#         shape = self.image.shape
#         if (image_shape := shape[-3:]) != self.image_shape:
#             self.coordinates = torch.meshgrid([torch.arange(s, device=image.device) for s in shape])
#             self.coordinates = torch.stack(self.coordinates, dim=-1).reshape(-1, 3)
#             self.image_shape = image_shape

#         if hemi is None:
#             hemi = self.hemispheres
#         else:
#             assert hemi in self.hemispheres
#             hemi = (hemi, )

#         hemi = hemi or self.hemispheres

#         features = self.unet(image)

#         features = self.head(features)



#         saliency = {}
#         weights = {}
#         predicted_points = {}
#         transform = {}

#         for h in hemi:
#             S = self.hemi_features(features).reshape(shape[0], shape[1], -1)
#             W = S.sum(-1)
#             X = S / W @ self.coordinates

#             # X @ vox_to_mri

#             transform[h] = self.WLS(X, self.template_points[h], W)

#             saliency[h] = S
#             weights[h] = W
#             predicted_points[h] = X

#         if len(hemi) == 2:
#             transform["brain"] = self.WLS(
#                 torch.cat(predicted_points.values()),
#                 torch.cat(self.template_points.values()),
#                 torch.cat(W.values()),
#             )

#         return transform