
import torch
import brainnet.mesh.topology
from brainnet.mesh.surface import TemplateSurfaces

import brainsynth
import nibabel as nib
from scipy.spatial import cKDTree


"""Find the 11 nearest triangles per vertex (measured with barycenters), or
rather, we enforce that the triangles of which a vertex is part are included
and then fill with the nearest triangles.
"""

n_topologies = 7

topologies = brainnet.mesh.topology.get_recursively_subdivided_topology(
            n_topologies - 1,
            brainnet.mesh.topology.initial_faces,
        )

top = topologies[-2]

sphere = TemplateSurfaces(
            torch.tensor(nib.freesurfer.read_geometry(brainsynth.resources_dir / "sphere-reg.srf")[0][:top.n_vertices],
                         dtype=torch.float), top)


indices, indptr, counts = sphere.topology.get_vertexwise_faces()



# 11 because the largest number of triangles that a vertex is part of is 11!
# (tensor([ 4,  5,  6,  7,  8, 11]),
#  tensor([     4,     15, 245737,      4,      1,      1]))
# (tensor([ 4,  5,  6,  7,  8, 11]),
#  tensor([    4,    15, 61417,     4,     1,     1]))

tree = cKDTree(sphere.compute_face_barycenters()[0])
dist, ind = tree.query(sphere.vertices.numpy()[0], 11)
ind = torch.tensor(ind)



all_faces = torch.zeros(sphere.topology.n_vertices, 11, dtype=torch.int)
for i in range(sphere.topology.n_vertices):
    c = counts[i]
    inds = indices[indptr[i]:indptr[i+1]]
    all_faces[i,:c] = inds
    all_faces[i, c:] = ind[i, ~torch.isin(ind[i], inds)][:11-c]

torch.save(all_faces, "brainnet/resources/template.5.nn_tri.pt")