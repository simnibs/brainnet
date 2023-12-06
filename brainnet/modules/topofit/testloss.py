import nibabel as nib
import torch
import pyvista as pv

from brainnet.mesh.topology import *
from brainnet.modules import loss



# gii = nib.load("/home/jesperdn/Downloads/data/m2m_100206/surfaces/lh.white.gii")
gii = nib.load("/mnt/scratch/INN/jesperdn/hcp_charm/m2m_100206/surfaces/lh.white.gii")
v, f = gii.agg_data()

vertices = torch.from_numpy(v)[None]
faces = torch.from_numpy(f).to(torch.int64)

# e.g.,
# smooth 3 times for resolution 5
# smooth 5 times for resolution 6
topology = Topology(faces)
v5 = topology.subsample_array(vertices[None].mT).mT

t5 = get_recursively_subdivided_topology(initial_faces, 5)[-1]




curv = compute_mean_curvature_vector(vertices[None], faces)
compute_iterative_spatial_smoothing(curv, topology, i).squeeze().numpy()

m = pv.make_tri_mesh(vertices.numpy().squeeze(), f)
m["curv00"] = curv.squeeze().numpy()
for i in range(1, 6):
    m[f"curv{i:02d}"] = compute_iterative_spatial_smoothing(curv, topology, i).squeeze().numpy()

m.save("/home/jesperdn/Downloads/data/mesh.vtk")

m.set_active_scalars("curvsmooth")
m.set_active_vectors("curv")

p = pv.Plotter(notebook=False)
p.add_mesh(m)
p.show()

curv_pred = compute_mean_curvature_vector(vertices[None], faces)
curv_true = compute_iterative_spatial_smoothing(curv_pred, topology, 3)

# dot product
a = curv_true.norm(dim=-1)
b = torch.sum(curv_pred*curv_true, -1)
loss = torch.mean((a-b)**2)

# element-wise MSE
loss = torch.nn.functional.mse_loss(curv_pred, curv_true, reduction="none")
loss = loss.mean(-1).squeeze().numpy()