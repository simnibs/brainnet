import torch

from brainnet.mesh.surface import Surface
from brainnet.mesh.topology import Topology

import nibabel as nib
import numpy as np

device = torch.device("cuda:0")


from brainnet.modules.losses_surface import VertexToVertexAngleLoss, FaceNormalConsistencyLoss
from brainnet.modules.losses import MSNELoss


class Decouple(torch.nn.Module):
    def __init__(self, n, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.h = torch.nn.Parameter(
            torch.zeros((1, n, 3), device=self.device))

    def forward(self, surface):
        # n = surface.compute_vertex_normals()
        # surface.vertices = surface.vertices + self.h * n
        # return surface
        # return self.h * n
        return self.h






from brainnet.modules.losses import SquaredCosineSimilarityError, SquaredNormError, AbsoluteError
from brainnet.modules.losses_surface import TriangleLengthVarianceLoss, EdgeLengthVarianceLoss, TaubinLoss

class SurfaceSmoothness(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sne = SquaredNormError()
        # self.sne = AbsoluteError()

    def forward(self, surface, values, mask=None):
        vi = values.index_select(1, surface.topology.conv_index_reduce)
        vj = values.index_select(1, surface.topology.conv_index_gather)
        if mask is None:
            return self.sne(vi,vj).mean()
            # return (1.0 - torch.sum(vi * vj, -1)).abs().mean()
        else:
            return torch.index_reduce(
                torch.zeros((surface.n_batch, surface.topology.n_vertices),
                            device=surface.device),
                1, surface.topology.conv_index_reduce.long(),
                self.sne(vi,vj),
                # (1.0-torch.sum(vi * vj, -1)).abs(),
                "mean",
                include_self=False,
            )[:,mask[0]].mean().nan_to_num()


class SignedDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cossim = torch.nn.CosineSimilarity(dim=-1)


    def forward(self, white, pial):
        vw = white.vertices
        vp = pial.vertices
        nw = white.compute_vertex_normals()
        cos = self.cossim(vp-vw, nw)
        dist = torch.linalg.vector_norm(pial.vertices-white.vertices, dim=-1)
        dist = dist * cos.sign()
        cutoff = 0.2
        self._mask =  dist < cutoff
        return (dist[self._mask] - cutoff).pow(2).mean()



loss_angle = VertexToVertexAngleLoss()
loss_smooth = FaceNormalConsistencyLoss()
loss_stationary = MSNELoss()
smoothness_loss = SurfaceSmoothness()
sdl = SignedDistanceLoss()

trivar = TriangleLengthVarianceLoss()
edgvar = EdgeLengthVarianceLoss()
taubin=TaubinLoss()


v,f,m = nib.freesurfer.read_geometry(
    "/home/jesperdn/repositories/brainsynth/brainsynth/resources/lh.white.smooth", read_metadata=True,
)


white = Surface(
    torch.tensor(v.astype(np.float32)[None], device=device),
    Topology(torch.tensor(f.astype(np.int32), device=device)))

model = Decouple(white.topology.n_vertices, device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

orig_vertices = white.vertices

for i in range(1,200+1):
    delta_v = model(white)
    white.vertices = white.vertices + delta_v
    # loss1 = loss_stationary(white.vertices, orig_vertices)
    loss1 = 100.0 * taubin(white)
    loss2 = 1.0 * edgvar(white)
    loss3 = 1.0 *trivar(white)

    loss = loss1 + loss2 + loss3
    if i % 10 == 0:
        print(f"Loss : {loss1:6.3f} + {loss2:6.3f} + {loss3:6.3f} = {loss:10.3f}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # pial.vertices = pial.vertices.detach()
    # delta_v = delta_v.detach()

nib.freesurfer.write_geometry(
    # "/mnt/projects/CORTECH/nobackup/jesper/recons_sharing/synthtopo_adni_t1s/225571/surf/lh.pial.opt",
    "test.opt",
    white.vertices.detach().cpu().numpy()[0], white.faces.cpu().numpy(),
    volume_info=m,
)


v,f = nib.freesurfer.read_geometry(
    # "/mnt/projects/CORTECH/nobackup/jesper/recons_sharing/synthtopo_adni_t1s/225571/surf/lh.white"
    "/mnt/projects/CORTECH/nobackup/exvivo/EXC022_new_confidence/surf/lh.white"
)
white = Surface(
    torch.tensor(v.astype(np.float32)[None], device=device),
    Topology(torch.tensor(f.astype(np.int32), device=device)))

v,f,m = nib.freesurfer.read_geometry(
    # "/mnt/projects/CORTECH/nobackup/jesper/recons_sharing/synthtopo_adni_t1s/225571/surf/lh.pial"
    "/mnt/projects/CORTECH/nobackup/exvivo/EXC022_new_confidence/surf/lh.pial",
    read_metadata=True
)
pial = Surface(
    torch.tensor(v.astype(np.float32)[None], device=device),
    Topology(torch.tensor(f.astype(np.int32), device=device)))


model = Decouple(pial.topology.n_vertices, device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

orig_vertices = pial.vertices

for i in range(1,200+1):
    delta_v = model(pial)
    pial.vertices = pial.vertices + delta_v
    loss0 = 10.0 * loss_angle(dict(white=white, pial=pial))
    # loss1 = loss_smooth(pial)
    loss1 = 1.0 * torch.sum((pial.vertices - orig_vertices)**2,-1).mean()
    loss2 = 1.0 * smoothness_loss(pial, delta_v)
    loss2 = 0.0 # sdl(white, pial)
    #dnorm = delta_v.norm(dim=-1)
    loss3 = 0.0 # 1.0 * smoothness_loss(pial, pial.vertices, sdl._mask)
    # loss3 = 0.0 # smoothness_loss(pial, pial.compute_vertex_normals(), sdl._mask)
    # loss3 = loss_smooth(pial)

    loss = loss0 + loss1 + loss2 + loss3
    if i % 10 == 0:
        print(f"Loss : {loss0:6.3f} + {loss1:6.3f} + {loss2:6.3f} + {loss3:6.3f} = {loss:10.3f}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # pial.vertices = pial.vertices.detach()
    # delta_v = delta_v.detach()

nib.freesurfer.write_geometry(
    # "/mnt/projects/CORTECH/nobackup/jesper/recons_sharing/synthtopo_adni_t1s/225571/surf/lh.pial.opt",
    "lh.pial.opt",
    pial.vertices.detach().cpu().numpy()[0], pial.faces.cpu().numpy(),
    volume_info=m,
)
