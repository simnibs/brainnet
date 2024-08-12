import copy
from pathlib import Path

import tqdm
import torch

import brainsynth
from brainnet.mesh.surface import TemplateSurfaces
from brainnet.mesh.topology import get_recursively_subdivided_topology

# rsync to /mnt/projects:
#
#   rsync -auvh -n /mnt/scratch/personal/jesperdn/decoupled /mnt/projects/CORTECH/nobackup/training_data

root_dir = Path("/mnt/projects/CORTECH/nobackup/")
out_dir = Path("/mnt/scratch/personal/jesperdn/decoupled/")

decouple_amount = 0.2 # mm

for res in (4, 5, 6):
    print(f"RESOLUTION {res}")

    surface_names = ("white", "pial")

    topology = get_recursively_subdivided_topology(res)[-1]
    topology = dict(lh=topology, rh=copy.deepcopy(topology))
    topology["rh"].reverse_face_orientation()

    templates = {
        h: {
            s: TemplateSurfaces(torch.zeros(t.n_vertices, 3), t)
            for s in surface_names
        }
        for h, t in topology.items()
    }

    data = brainsynth.config.DatasetConfig(
        root_dir = root_dir / "training_data",
        subject_dir = root_dir / "training_data_subjects",
        subject_subset = None, # train.subset-0020
        # synthesizer = SynthesizerConfig( ... ),
        # datasets = ["ABIDE", "HCP"], # default: all
        # images = ["generation_labels", "t1w"],
        images = [],
        # ds_structure = "flat",
        target_surface_resolution = res,
    )

    for k,v in data.dataset_kwargs.items():
        print(f"Dataset {k}")
        ds = brainsynth.dataset.SynthesizedDataset(**v)
        for i in tqdm.tqdm(range(len(ds))):
            surf_dir = out_dir / f"{ds.name}.{ds.subjects[i]}.surf_dir"
            if not surf_dir.exists():
                surf_dir.mkdir()

            _, surfaces, _ = ds[i]
            for h,x in surfaces.items():
                pial = templates[h]["pial"]
                pial.vertices = x["pial"]

                zero = torch.norm(x["pial"] - x["white"], dim=1) < 1e-3
                # white and pial normals should be almost identical at these nodes
                normals = pial.compute_vertex_normals()

                surfaces[h]["pial"][zero] += decouple_amount * normals.squeeze()[zero]

            for h,x in surfaces.items():
                # pial
                f = surf_dir/ f"{h}.pial.{res}.target-decoupled.pt"
                torch.save(x["pial"], f)

                # just symlink to white as we only modify pial
                f = surf_dir / f"{h}.white.{res}.target-decoupled.pt"
                f_to = ds.ds_dir / f"{ds.name}.{ds.subjects[i]}.surf_dir" / f"{h}.white.{res}.target.pt"
                f.symlink_to(f_to)

            # for h,x in surfaces.items():

            #     nib.freesurfer.write_geometry(
            #         ds.ds_dir / f"{ds.name}.{ds.subjects[i]}.surf_dir" / f"pialNOdecoupled{h}",
            #         x["pial"].numpy(),
            #         templates[h]["pial"].faces.numpy()
            #     )

            #     nib.freesurfer.write_geometry(
            #         ds.ds_dir / f"{ds.name}.{ds.subjects[i]}.surf_dir" / f"whiteNOdecoupled{h}",
            #         x["white"].numpy(),
            #         templates[h]["white"].faces.numpy()
            #     )

