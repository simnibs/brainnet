from pathlib import Path
import torch
#from brainnet.train import BrainNetTrainer, load_config
import numpy as np
import nibabel as nib

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

def get_affine_mni_to_mri(m2m):
    """Affine transformation from MNI to subject MRI coordinates."""


    # template = nib.load(
    #     Path(simnibs.utils.file_finder.templates.charm_atlas_path)
    #     / "charm_atlas_mni"
    #     / "template.nii"
    # )
    template = nib.load(
        "/mrhome/jesperdn/repositories/simnibs/simnibs/segmentation/atlases/charm_atlas_mni/template.nii"
    )
    # template_coreg = nib.load(m2m.template_coregistered)
    template_coreg = nib.load(
        "/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/segmentation/template_coregistered.mgz"
    )
    return template_coreg.affine @ np.linalg.inv(template.affine)


def compute_mni305_to_vox(m2m, vox_to_mri):
    mni152_to_mri = get_affine_mni_to_mri(m2m)
    mri_to_vox = np.linalg.inv(vox_to_mri)
    return mri_to_vox @ mni152_to_mri @ mni305_to_mni152

from brainsynth.prepare_freesurfer_data import align_with_identity_affine
import surfa
from nibabel.affines import apply_affine
from brainsynth import root_dir


import brainnet
from brainnet import config
from brainnet.modules import body, head

from brainnet.mesh.topology import get_recursively_subdivided_topology
tops = get_recursively_subdivided_topology(5)
top = tops[-1]
faces = top.faces
top0 = tops[0]


f = top0.faces
# m2m = "/home/jesperdn/nobackup/simnibs4_examples/m2m_ernie"
img = nib.load("/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/T2_reg.nii.gz")
img = align_with_identity_affine(img)
mni305_to_vox = compute_mni305_to_vox(None, img.affine)

white_lh_template = surfa.load_mesh(str(root_dir / "resources" / "cortex-int-lh.srf"))
l2r = surfa.load_affine(str(root_dir / "resources" / "left-to-right.lta"))

template_mesh = dict(lh=white_lh_template, rh=white_lh_template.transform(l2r))

full = apply_affine(mni305_to_vox, template_mesh["lh"].vertices)

center = 0.5 * (full.max(0) - full.min(0))

init_verts = apply_affine(mni305_to_vox, template_mesh["lh"].vertices[:62])

image = torch.from_numpy(img.get_fdata()).float()[None].cuda()
init_verts = torch.from_numpy(init_verts).float().cuda()

x = apply_affine(img.affine, init_verts.cpu().numpy()).astype(np.float32)

gii = nib.GiftiImage(darrays=(
    nib.gifti.GiftiDataArray(x, "pointset"),
    nib.gifti.GiftiDataArray(f.cpu().numpy(), "triangle"),
    )
)
gii.to_filename("/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/initverts.gii")

# img.to_filename("/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/T1_aligned.nii")


device = torch.device("cuda")


size = torch.tensor([128.0, 224.0, 160.0]).to(device)

c = init_verts.mean(0)
lo = c - size/2.0
lo = lo.floor()
hi = c + size/2.0
hi = hi.ceil()

hi += size - (hi-lo)

cor = torch.maximum(-lo, torch.zeros_like(lo))
lo += cor
hi += cor

slimg = img.slicer[*[slice(l.int().item(),h.int().item()) for l,h in zip(lo, hi)]]
image = torch.from_numpy(slimg.get_fdata()).float()[None].cuda()

slaffine = slimg.affine

init_verts = apply_affine(np.linalg.inv(slaffine), x)




spatial_dims = 3
in_channels = 1
unet_enc_ch = [[32], [64], [96], [128], [160]]
unet_dec_ch = [[128], [96], [64], [64]]
unet_out_ch = unet_dec_ch[-1][-1]

model_config = config.BrainNetParameters(
    device=device,
    body = body.UNet(spatial_dims, in_channels, unet_enc_ch, unet_dec_ch),
    heads = dict(
        surface = head.SurfaceModule(
            in_channels=unet_out_ch,
            prediction_res=5,
            device=device,
        ),
#                 kwargs = dict(prediction_res = target_surface_resolution),

        # segmentation = SegmentationModule(...)
    ),
)
model = brainnet.BrainNet(model_config.body, model_config.heads, device)
model.to(device)

from ignite.handlers import ModelCheckpoint


to_load = dict(model=model)


ckpt_name = "state_checkpoint_02000.pt"
ckpt = Path("/mnt/scratch/personal/jesperdn/results/BrainNet/lh-01-1mm_iso/checkpoint/") / ckpt_name
ModelCheckpoint.load_objects(to_load, ckpt)

ql = image.quantile(0.001)
qu = image.quantile(0.999)
image = torch.clip((image - ql) / (qu - ql), 0.0, 1.0)

# out_size = [128, 224, 160]
# out_center_str = "lh"

# config = brainsynth.config.SynthesizerConfig(
#     builder = "OnlySelect",
#     out_size = out_size,
#     out_center_str = out_center_str,
#     alternative_images = ["t1w"],
#     device = torch.device("cuda:0"),
# )
# synth = brainsynth.Synthesizer(config)

with torch.inference_mode():
    y_pred = model(image[None], dict(lh=torch.from_numpy(init_verts[None]).float().to(device)))



for k,v in y_pred["surface"]["lh"].items():

    gii = nib.GiftiImage(darrays=(
        nib.gifti.GiftiDataArray(apply_affine(slimg.affine.astype(np.float32), v[0].cpu().numpy()), "pointset"),
        nib.gifti.GiftiDataArray(faces.cpu().numpy(), "triangle"),
        )
    )
    gii.to_filename(f"/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/lh.1mmiso.pred.T2.{k}.gii")


    # gii = nib.GiftiImage(darrays=(
    #     nib.gifti.GiftiDataArray(apply_affine(slimg.affine.astype(np.float32), y_true1["lh"][k][0].cpu().numpy()), "pointset"),
    #     nib.gifti.GiftiDataArray(self.model.heads.surface.topologies[5].faces.cpu().numpy(), "triangle"),
    #     )
    # )
    # gii.to_filename(f"/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/T13000_lh.{k}.true.T1.gii")



    gii = nib.GiftiImage(darrays=(
        nib.gifti.GiftiDataArray(apply_affine(slimg.affine.astype(np.float32), y_pred2["surface"]["lh"][k][0].cpu().numpy()), "pointset"),
        nib.gifti.GiftiDataArray(self.model.heads.surface.topologies[5].faces.cpu().numpy(), "triangle"),
        )
    )
    gii.to_filename(f"/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/adaptT1w_lh.{k}.pred.T1.gii")






###########




ckpt = 2600

config = load_config("/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_foundation.yaml")
trainer = BrainNetTrainer(config)
trainer.load_checkpoint(ckpt)



n = len(trainer.dataloaders["train"].dataset)
for i in range(n):
    subdir, ds_id, y_true_img, y_true_surf, init_vertices, info = trainer.dataloaders["train"].dataset[i]

    image = y_true_img["T1"][None].cuda()
    init_verts = {k: v[None].cuda() for k,v in init_vertices.items()}

    print(f"{i:4d} / {n:4d} : {subdir}")
    with torch.inference_mode():
        # image, y_true, init_verts = trainer.apply_synthesizer(
        #     y_true_img, y_true_surf, init_vertices, info, ds_id, disable_synth=True
        # )
        y_pred = trainer.model(image, init_verts)
    for h,v in y_pred["surface"].items():
        for s, vv in v.items():
            torch.save(vv[0].cpu(), subdir / f"{h}.{s}.5.prediction.pt")

n = len(trainer.dataloaders["validation"].dataset)
for i in range(n):
    subdir, ds_id, y_true_img, y_true_surf, init_vertices, info = trainer.dataloaders["validation"].dataset[i]

    image = y_true_img["T1"][None].cuda()
    init_verts = {k: v[None].cuda() for k,v in init_vertices.items()}

    print(f"{i:4d} / {n:4d} : {subdir}")
    with torch.inference_mode():
        # image, y_true, init_verts = trainer.apply_synthesizer(
        #     y_true_img, y_true_surf, init_vertices, info, ds_id, disable_synth=True
        # )
        y_pred = trainer.model(image, init_verts)
    for h,v in y_pred["surface"].items():
        for s, vv in v.items():
            torch.save(vv[0].cpu(), subdir / f"{h}.{s}.5.prediction.pt")

nib.freesurfer.write_geometry(subdir / f"{h}.{s}.5.prediction.freesurfer", vv[0].cpu().numpy(), faces.numpy())
nib.Nifti1Image(image.cpu().numpy().squeeze(), np.eye(4)).to_filename(subdir / f"test_image.nii")

if __name__ == "__main__":

    config = load_config(
        "/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_foundation.yaml"
    )
    self = BrainNetTrainer(config)
    self.examples_dir = Path("/home/jesperdn/nobackup/brainnet_eval")
    if not self.examples_dir.exists():
        self.examples_dir.mkdir()

    ds_id, images, surfaces, temp_verts, info = next(iter(self.dataloaders["validation"]))
    image, y_true, init_verts = self.apply_synthesizer(images, surfaces, temp_verts, info, ds_id[0], disable_synth=True)

    ckpts = (200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2200, 2400, 2600)

    for ckpt in ckpts:

        self.load_checkpoint(ckpt)
        with torch.inference_mode():
            y_pred = self.model(image, init_verts, head_kwargs=self.head_runtime_kwargs)

        # convert surface predictions to batched surfaces
        if (k := "surface") in y_pred:
            # insert vertices into template surface
            self.set_templatesurface(y_pred[k], self.surface_skeletons["y_pred"])


            # self.set_templatesurface(y_true_out[k], self.surface_skeletons["y_true"])

            # self.criterion.precompute_for_surface_loss(y_pred[k], y_true[k])
            # self.criterion.prepare_for_surface_loss(y_pred[k], y_true[k])

        with torch.inference_mode():
            features = self.model.body(image)

        nib.Nifti1Image(features[0].detach().cpu().numpy().transpose(1,2,3,0), np.identity(4)).to_filename(
            "/home/jesperdn/nobackup/brainnet_eval/features.nii"
        )
        nib.Nifti1Image(image[0,0].detach().cpu().numpy(), np.identity(4)).to_filename("/home/jesperdn/nobackup/brainnet_eval/norm.nii")


        # y_pred = y_pred["surface"]["lh"]["white"]
        # y_true = y_true["surface"]["lh"]["white"]

        # ip = y_pred.nearest_neighbor(y_true)
        # it = y_true.nearest_neighbor(y_pred)


        # d1 = torch.sum((y_pred.vertices[0] - y_true.vertices[0,ip[0]])**2, -1)
        # d2 = torch.sum((y_pred.vertices[0,it[0]] - y_true.vertices[0])**2, -1)


        # dd1 = torch.sum((y_pred.vertices[0] - y_true.vertices[0,ip[0]])**2, -1).sqrt()
        # dd2 = torch.sum((y_pred.vertices[0,it[0]] - y_true.vertices[0])**2, -1).sqrt()


        # f = y_pred.faces.cpu().numpy()
        # m = pv.make_tri_mesh(y_pred.vertices[0].cpu().numpy(), f)
        # m["chamfer"] = d1.cpu().numpy()
        # m.save("/home/jesperdn/nobackup/pred.vtk")
        # m = pv.make_tri_mesh(y_true.vertices[0].cpu().numpy(), f)
        # m["chamfer"] = d2.cpu().numpy()
        # m.save("/home/jesperdn/nobackup/true.vtk")

        # loss = self.criterion(y_pred, y_true)
        prefix= "eval"
        n = ckpt
        self.write_results(prefix, n, image, y_pred, y_true, init_verts)


m = pv.make_tri_mesh(y_true["surface"]["lh"]["pial"].vertices.cpu().numpy()[0],
                     y_true["surface"]["lh"]["pial"].faces.cpu().numpy()
                     )
m["intersecting faces"] = a.cpu().numpy()
m.save("/home/jesperdn/nobackup/brainnet_eval/selfintersections.vtk")


print("Self-intersections")
print()
print("pred              true")
print("white    pial     white   pial     ")
fmt = "{:<8d} {:<8d} {:<8d} {:<8d}"

for i,(ds_id, images, surfaces, temp_verts, info) in enumerate(self.dataloaders["validation"]):
    # if i < 200:
        # continue
    image, y_true, init_verts = self.apply_synthesizer(images, surfaces, temp_verts, info, ds_id[0], disable_synth=True)
    with torch.inference_mode():
        y_pred = self.model(image, init_verts, head_kwargs=self.head_runtime_kwargs)

    # convert surface predictions to batched surfaces
    if (k := "surface") in y_pred:
        # insert vertices into template surface
        self.set_templatesurface(y_pred[k], self.surface_skeletons["y_pred"])


    print(fmt.format(
        y_pred["surface"]["lh"]["white"].compute_self_intersections()[1].item(),
        y_pred["surface"]["lh"]["pial"].compute_self_intersections()[1].item(),
        y_true["surface"]["lh"]["white"].compute_self_intersections()[1].item(),
        y_true["surface"]["lh"]["pial"].compute_self_intersections()[1].item(),
    ))

    a,b = y_pred["surface"]["lh"]["white"].compute_self_intersections()

    if b.cpu().item() > 40:

        m = pv.make_tri_mesh(y_pred["surface"]["lh"]["white"].vertices.cpu().numpy()[0],
                            y_pred["surface"]["lh"]["white"].faces.cpu().numpy()
                            )
        m["intersecting faces"] = a.cpu().numpy()
        m.save("/home/jesperdn/nobackup/brainnet_eval/selfintersections.vtk")



lh = nib.load("/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/surfaces/lh.pial.gii")
v,f=lh.agg_data()
v = v @ np.linalg.inv(img.affine[:3,:3]).T - img.affine[:3,3]






image1, y_true1, init_verts1 = s(
    dict(norm=monai.data.MetaTensor(image, img.affine)[0],
         generation=monai.data.MetaTensor(image, img.affine)[0]),
    dict(lh=dict(white=monai.data.MetaTensor(init_verts)[0])),
    dict(lh=monai.data.MetaTensor(init_verts)[0]),
    info,
    disable_synth=True
)
slimg = img.slicer[*s.spatial_crop.slices]

for k in image1:
    image1[k] = s.normalize_intensity(image1[k])

with torch.inference_mode():
    y_pred = self.model(image1["norm"][None], {k:v[None] for k,v in init_verts1.items()})

y_pred_full = {k:{kk:vv + torch.tensor([7.0, 17.0, 86.0])[None].cuda() for kk,vv in v.items()} for k,v in y_pred["surface"].items()}

config = load_config(
    "/mrhome/jesperdn/repositories/brainnet/brainnet/config/train_foundation_t1w.yaml"
)
trainer = BrainNetTrainer(config)
trainer.load_checkpoint(800)

with torch.inference_mode():
    y_pred2 = trainer.model(image1["norm"][None], y_pred["surface"])


for k,v in y_pred["surface"]["lh"].items():

    gii = nib.GiftiImage(darrays=(
        nib.gifti.GiftiDataArray(apply_affine(slimg.affine.astype(np.float32), v[0].cpu().numpy()), "pointset"),
        nib.gifti.GiftiDataArray(self.model.heads.surface.topologies[5].faces.cpu().numpy(), "triangle"),
        )
    )
    gii.to_filename(f"/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/T13000_lh.{k}.pred.T1.gii")

    gii = nib.GiftiImage(darrays=(
        nib.gifti.GiftiDataArray(apply_affine(slimg.affine.astype(np.float32), y_true1["lh"][k][0].cpu().numpy()), "pointset"),
        nib.gifti.GiftiDataArray(self.model.heads.surface.topologies[5].faces.cpu().numpy(), "triangle"),
        )
    )
    gii.to_filename(f"/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/T13000_lh.{k}.true.T1.gii")



    gii = nib.GiftiImage(darrays=(
        nib.gifti.GiftiDataArray(apply_affine(slimg.affine.astype(np.float32), y_pred2["surface"]["lh"][k][0].cpu().numpy()), "pointset"),
        nib.gifti.GiftiDataArray(self.model.heads.surface.topologies[5].faces.cpu().numpy(), "triangle"),
        )
    )
    gii.to_filename(f"/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/adaptT1w_lh.{k}.pred.T1.gii")


nib.Nifti1Image(
    image1["norm"].detach().cpu().numpy().squeeze(),
    np.identity(4)
).to_filename("/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/t1cropped.nii")

gii = nib.load("/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/surfaces/lh.white.gii")


v,f = nib.load("/home/jesperdn/nobackup/ernie/m2m_ernie_charm_fs/new2200_lh.white.pred.T1.gii").agg_data()

# import simnibs
# import simnibs.utils.file_finder

