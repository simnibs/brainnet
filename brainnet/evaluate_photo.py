from pathlib import Path
import torch
from brainnet.train import BrainNetTrainer, load_config
import numpy as np
import nibabel as nib
import nibabel.processing

root = Path("/home/jesperdn/nobackup/photo")
img = nib.load(root / "native_ras.nii.gz")
# flimg = img.slicer[::-1]
flimg = nib.Nifti1Image(img.get_fdata()[::-1], img.affine)
flimg.to_filename(root / "native_ras_lr_flip.nii.gz")

for i in range(3):
    nibabel.processing.resample_to_output(
        nib.Nifti1Image(img.get_fdata()[::-1,...,i], img.affine), 1, order=1
    ).to_filename(root / f"native_ras_lr_flip_1mm_{i}.nii.gz")


ckpt = 4200

config = load_config(
    "/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_foundation.yaml"
)
trainer = BrainNetTrainer(config)

state = torch.load(f"/mnt/scratch/personal/jesperdn/results/BrainNetEdgeNet/SingleHemi2-PHOTO/checkpoints/state_0{ckpt}.pt")
trainer.model.load_state_dict(state["model_state"], strict=False)

from brainsynth import root_dir
from brainsynth.prepare_freesurfer_data import align_with_identity_affine
import surfa
from nibabel.affines import apply_affine
from brainsynth import Synthesizer
import monai

f = trainer.model.heads.surface.topologies[0].faces
img = nib.load("/home/jesperdn/nobackup/photo/native_ras_lr_flip_1mm_2.nii.gz")
# img = nib.load("/home/jesperdn/nobackup/photo/mni_t1.nii.gz")

aff = np.loadtxt("/home/jesperdn/nobackup/photo/mni_t1.txt")
mni = nib.load("/home/jesperdn/nobackup/photo/MNI152_lh.nii.gz")



# inverse of 8b from
# https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
mni305_to_mni152 = np.array(
    [
        [ 0.9975,     -0.0073,      0.0176,     -0.0429    ],
        [ 0.0146,      1.00090003, -0.0024,      1.54960001],
        [-0.013,      -0.0093,      0.9971,      1.18400002],
        [ 0.,          0.,          0.,          1.     ],
    ]
)

mni305_to_vox = np.linalg.inv(img.affine) @ aff @ mni305_to_mni152

white_lh_template = surfa.load_mesh(str(root_dir / "resources" / "cortex-int-lh.srf"))
l2r = surfa.load_affine(str(root_dir / "resources" / "left-to-right.lta"))

template_mesh = dict(lh=white_lh_template, rh=white_lh_template.transform(l2r))

full = apply_affine(mni305_to_vox, template_mesh["lh"].vertices)

center = 0.5 * (full.max(0) - full.min(0))

init_verts = apply_affine(mni305_to_vox, template_mesh["lh"].vertices[:62])

image = torch.from_numpy(img.get_fdata()).float()[None, None].cuda()
init_verts = torch.from_numpy(init_verts).float()[None].cuda()


gii = nib.GiftiImage(darrays=(
    nib.gifti.GiftiDataArray(apply_affine(img.affine, init_verts[0].cpu().numpy()).astype(np.float32), "pointset"),
    nib.gifti.GiftiDataArray(f.cpu().numpy(), "triangle"),
    )
)
gii.to_filename("/home/jesperdn/nobackup/photo/init_vertices.gii")


info = dict(bbox=dict(lh=torch.tensor(np.array([full.min(0), full.max(0)])).cuda()),
            shape=torch.tensor(img.shape).cuda(),
            resolution=torch.tensor(img.header.get_zooms()).cuda())
s = Synthesizer(config="synthesizer_no_synth.yaml", device=torch.device("cuda:0"))
s.config.intensity.probability=0


image = s.normalize_intensity(image)
image_pad = torch.zeros((1,1,128,192,176)).to(image.device)
image_pad[:, :, :image.shape[2], :image.shape[3], :image.shape[4]] = image

with torch.inference_mode():
    y_pred = trainer.model(image_pad, dict(lh=init_verts))



for k,v in y_pred["surface"]["lh"].items():
    gii = nib.GiftiImage(darrays=(
        nib.gifti.GiftiDataArray(apply_affine(img.affine.astype(np.float32), v[0].cpu().numpy()), "pointset"),
        nib.gifti.GiftiDataArray(trainer.model.heads.surface.topologies[5].faces.cpu().numpy(), "triangle"),
        )
    )
    gii.to_filename(root / f"photo_{ckpt}_lh.{k}.pred.gii")
