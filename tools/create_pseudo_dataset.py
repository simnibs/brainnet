import sys

import shutil
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.processing
import torch
from torch.utils.data import default_collate

import brainsynth.config

from brainnet import config

import brainnet.train.utilities

import brainnet.initializers

from brainnet.config.brainreg.main import train_setup


from brainsynth.transforms import (
    EnsureDevice,
    IntensityNormalization,
)

root_dir: Path = Path("/mnt/projects/CORTECH/nobackup/training_data")


target_surface_resolution: int = 5
initial_surface_resolution: int = 0

target_surface_hemisphere: str = "both"

train_setup.train_params.load_checkpoint = 300
train_setup.wandb.enable = False

criterion = brainnet.initializers.init_criterion(train_setup.criterion)
dataloader = brainnet.initializers.init_dataloader(
    train_setup.dataset, train_setup.dataloader
)
model = brainnet.initializers.init_model(train_setup.model)
optimizer = brainnet.initializers.init_optimizer(train_setup.optimizer, model)
synth = brainnet.initializers.init_synthesizer(train_setup.synthesizer)

to_save = dict(
    model=model,
)
brainnet.train.utilities.load_checkpoint(to_save, train_setup)

cfg_dataloader = config.DataloaderParameters()
cfg_dataset = config.DatasetParameters(
    train=brainsynth.config.DatasetConfig(
        root_dir=root_dir / "brainreg",
        subject_dir="/mnt/scratch/personal/jesperdn/training_data",
        subject_subset="source1",
        images=["t1w_areg_mni"],
        datasets = ["HCP"],
        load_mask="force",
        target_surface_resolution=target_surface_resolution,
        target_surface_hemispheres=target_surface_hemisphere,
        # target_surface_files=SURFACE.files.target,
        initial_surface_resolution=initial_surface_resolution,
    ),
    validation=brainsynth.config.DatasetConfig(
        root_dir=root_dir / "brainreg",
        subject_dir="/mnt/scratch/personal/jesperdn/training_data",
        subject_subset="target1",
        images=["t1w_areg_mni"],
        datasets = ["HCP"],
        load_mask="force",
        target_surface_resolution=target_surface_resolution,
        target_surface_hemispheres=target_surface_hemisphere,
        # target_surface_files=SURFACE.files.target,
        initial_surface_resolution=initial_surface_resolution,
    ),
)

dataloader = brainnet.initializers.init_dataloader(cfg_dataset, cfg_dataloader)

surface_resolution = {
    k: next(iter(v.dataset_kwargs.values()))["target_surface_resolution"]
    for k, v in vars(train_setup.dataset).items()
}

ensure_device = EnsureDevice(model.device)
intensity_normalization = IntensityNormalization(device=model.device)


def prepare_batch(batch, ensure_device, intensity_normalization):

    images, surfaces, initverts = batch

    images = ensure_device(images)
    surfaces = ensure_device(surfaces)
    initverts = ensure_device(initverts)

    assert (
        len(next(iter(images))) % 2 == 0
    ), "Even number of examples in a batch required."

    # We abuse the batching done by the dataloader and reshape like
    #
    #   batch = (N,1,W,H,D) -> (N/2,2,W,H,D)
    #
    # such that we can use consecutive subjects are registration pairs.
    # for k, v in images.items():
    #     size = v.size()
    #     images[k] = v.reshape(size[0] // 2, 2, *size[2:])

    # for k, v in surfaces.items():
    #     for kk, vv in v.items():
    #         size = vv.size()
    #         surfaces[k][kk] = vv[None].reshape(size[0] // 2, 2, *size[2:])

    # assume synthesizer was applied when loading the data

    for k, v in images.items():
        if v.is_floating_point():
            images[k] = intensity_normalization(v)
    if len(surfaces) > 0:
        images["surface"] = surfaces
        images["initial_vertices"] = initverts
    return images["t1w_areg_mni"], images


def predict_source_to_target(model, images, y_true):
    # batch (N, 1, ...) was reshaped as (N/2, 2, ...) thus
    # y_pred = (N/2, 3, ...)
    y_pred = dict(svf=model(images))

    # deform 0 to 1

    # we swap 0 and 1 when returning the predicted image(s) and surfaces.
    # This way, they can be directly compared with y_true

    # y_true has [sub-01, sub-02] in channel dim but y_pred is returned
    # as [deformed(sub-01), deformed(sub-02)] in channel dim so we need
    # to swap either y_pred or y_true when we calculate the losses
    # (such that deformed(sub-01) aligns with sub-02 etc.)

    deform_fwd, deform_bwd = model.integrate_svf(y_pred["svf"])

    # stack predictions inversely:
    # align the prediction of 0 to image 1 and prediction of 1 to image 1
    for k0, v0 in y_true.items():
        if k0 == "surface":
            y_pred[k0] = {}
            for k1, v1 in v0.items():  # hemi
                y_pred[k0][k1] = {}
                for k2, v2 in v1.items():  # surfaces
                    s0 = model.deform_surface(v2, deform_bwd)
                    y_pred[k0][k1][k2] = s0
        elif k0 == "initial_vertices":
            y_pred[k0] = {}
            for k1, v1 in v0.items():  # hemi
                s0 = model.deform_surface(v1, deform_bwd)
                y_pred[k0][k1] = s0
        else:
            i0 = model.deform_image(v0, deform_fwd)
            y_pred[k0] = i0
    return y_pred

def get_dataset_and_subject_from_index(index, concat_ds):
    for i,cs in enumerate(concat_ds.cumulative_sizes):
        if index < cs:
            ds = concat_ds.datasets[i]
            if i > 0:
                sub_idx = index - concat_ds.cumulative_sizes[i-1]
            else:
                sub_idx = index
            break

    subject = ds.subjects[sub_idx]
    return ds.name, subject


def copy_data(sub_dir_target, sub_out_dir):
    if not sub_out_dir.exists():
        sub_out_dir.mkdir(parents=True)

    shutil.copy(sub_dir_target / "T1w.nii", sub_out_dir)
    if (sub_dir_target / "T1w.defacingmask.nii").exists():
        shutil.copy(
            sub_dir_target / "T1w.defacingmask.nii",
            sub_out_dir
        )
    # shutil.copy(sub_dir_target / "generation_labels_dist.nii", sub_out_dir)

    for h in ("lh", "rh"):
        for s in ("white", "pial"):
            shutil.copy(sub_dir_target / f"{h}.{s}.5.target-decoupled.pt", sub_out_dir)
        shutil.copy(sub_dir_target / f"{h}.0.template.pt", sub_out_dir)


from brainnet.mesh.topology import get_recursively_subdivided_topology
topologies = get_recursively_subdivided_topology(5)
faces0 = topologies[0].faces.numpy().astype(np.int32)
faces5 = topologies[5].faces.numpy().astype(np.int32)

def write_data(y, vox2mni, mni2ras, target_img, sub_out_dir):
    if not sub_out_dir.exists():
        sub_out_dir.mkdir(parents=True)

    target_tuple = (target_img.shape[:3], target_img.affine)
    sub_ras2vox = np.linalg.inv(target_img.affine)

    # T1w
    out = nib.Nifti1Image(
        (255*y["t1w_areg_mni"]).cpu().numpy().squeeze().astype(np.uint8),
        mni2ras @ vox2mni
    )
    out = nibabel.processing.resample_from_to(out, target_tuple)
    out.to_filename(sub_out_dir / "T1w.nii")

    # mask
    out = nib.Nifti1Image(
        y["t1w_areg_mni_mask"].cpu().numpy().squeeze().astype(np.uint8),
        mni2ras @ vox2mni
    )
    out = nibabel.processing.resample_from_to(out, target_tuple)
    out.to_filename(sub_out_dir / "T1w.defacingmask.nii")

    # generation
    # out = nib.Nifti1Image(
    #     y["generation_labels_dist"].cpu().numpy().squeeze().astype(np.uint8),
    #     mni2ras @ vox2mni
    # )
    # out = nibabel.processing.resample_from_to(out, target_tuple, order=0)
    # out.to_filename(sub_out_dir / "generation_labels_dist.nii")

    vox2vox = torch.tensor(sub_ras2vox @ mni2ras @ vox2mni).float()

    # surfaces
    for h,v in y["surface"].items():
        for s,vv in v.items():
            name = ".".join([h,s,"5","target-decoupled","pt"])

            data = vv.cpu()
            out = data @ vox2vox[:3,:3].T + vox2vox[:3,3]
            torch.save(out, sub_out_dir / name)

            # nib.freesurfer.write_geometry(
            #     sub_out_dir / f"{h}.{s}.5.target-decoupled",
            #     (out.numpy() @ target_img.affine[:3,:3].T + target_img.affine[:3, 3]).astype(np.float32),
            #     faces5,
            # )

    # template
    for h,v in y["initial_vertices"].items():
        name = ".".join([h,"0","template","pt"])

        data = v.cpu()
        out = data @ vox2vox[:3,:3].T + vox2vox[:3,3]
        torch.save(out, sub_out_dir / name)

        # nib.freesurfer.write_geometry(
        #     sub_out_dir / f"{h}.0.template",
        #     (out.numpy() @ target_img.affine[:3,:3].T + target_img.affine[:3, 3]).astype(np.float32),
        #     faces0,
        # )

"""
#generate source/target subject splits


import numpy as np

subjects = np.genfromtxt("/mnt/projects/CORTECH/nobackup/training_data/subject_splits/HCP.train.txt", dtype=str)

sub_perm = np.random.permutation(subjects)

n = 1
m = len(sub_perm)-n

source = sub_perm[:n]
target = sub_perm[n:]

np.savetxt(
    "/mnt/scratch/personal/jesperdn/training_data/HCP.source1.txt",
    source.tolist(), fmt="%s"
)

np.savetxt(
    "/mnt/scratch/personal/jesperdn/training_data/HCP.target1.txt",
    target.tolist(), fmt="%s"
)

"""

if __name__ == "__main__":

    index = int(sys.argv[1])

    out_dir = Path("/mnt/scratch/personal/jesperdn/training_data/pseudo1")

    base_dir = root_dir / "brainreg"
    base_dir_target = root_dir / "full"

    dataset_source = dataloader["train"].dataset
    dataset_target = dataloader["validation"].dataset

    n = len(dataset_source)
    m = len(dataset_target)
    m_per_sub = m//n
    remainder = m - m_per_sub * n

    assert index < n


    model.eval()

    # i = subjects_source[index]

    ds_source, sub_source = get_dataset_and_subject_from_index(index, dataset_source)

    sub_in_dir = base_dir / ds_source / sub_source
    sub_out_dir = out_dir / ds_source / sub_source
    sub_dir_target = base_dir_target / ds_source / sub_source

    # COPY
    print(f"Copying {ds_source}/{sub_source}", flush=True)

    copy_data(sub_dir_target, sub_out_dir)

    batch_source = dataloader["train"].dataset[index]
    _, y_true_source = prepare_batch(batch_source, ensure_device, intensity_normalization)

    # SOURCE -> TARGET
    # for j in subjects_target[index]:
    start = index * m_per_sub
    stop = m if (index + 1) == n else (index + 1) * m_per_sub
    print(f"using target subject indices: {start} - {stop}")
    for j in range(start, stop):

        ds_target, sub_target = get_dataset_and_subject_from_index(j, dataset_target)

        print(f"{ds_source}/{sub_source} -> {ds_target}/{sub_target}", flush=True)

        sub_in_dir = base_dir / ds_target / sub_target
        sub_out_dir = out_dir / ds_target / sub_target
        sub_dir_target = base_dir_target / ds_target / sub_target

        mni2ras = np.loadtxt(sub_dir_target / "mni152_affine_forward.txt")
        vox2mni = nib.load(sub_in_dir / "T1w.areg-mni.nii").affine
        target_img = nib.load(sub_dir_target / "T1w.nii")

        batch_target = dataloader["validation"].dataset[j]
        _, y_true_target = prepare_batch(batch_target, ensure_device, intensity_normalization)

        batch = default_collate([batch_source, batch_target])
        images, y_true = prepare_batch(batch, ensure_device, intensity_normalization)

        with torch.inference_mode():
            with torch.autocast(model.device.type, enabled=train_setup.train_params.enable_amp):
                y_pred = predict_source_to_target(model, images, y_true_source)

        write_data(y_pred, vox2mni, mni2ras, target_img, sub_out_dir)

