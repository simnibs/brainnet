import argparse
import csv
import importlib
import itertools
import sys

import tqdm

import brainsynth
from brainsynth.config.synthesizer import PredictionConfig

from brainnet.event_handlers import FREESURFER_VOLUME_INFO, write_surfaces
import brainnet.train.utilities
from brainnet.evaluation.brainnet_eval import get_setup_at_checkpoint
from brainnet.prediction.brainnet_predict import PredictionStep


def create_predictor(model, setup, subset: str = "validation"):
    train = importlib.import_module(f".{model}", "brainnet.train")

    model = train.setup_model(setup)

    # We need the affine as well
    for v in setup.dataset[args.subset].dataset_kwargs.values():
        v["return_affine"] = True

    dataloaders = brainsynth.dataset.setup_dataloader(
        setup.dataset[args.subset],
        separate_datasets=True,
        **setup.dataloader,
    )
    preprocessor = brainsynth.Synthesizer(
        PredictionConfig(
            "PredictionBuilder",
            setup.synthesizer[subset].out_size,
            setup.synthesizer[subset].out_center_str,
        )
    )

    pred_step = train.PredictionStep(preprocessor, model, setup.enable_amp)

    to_load = dict(model=model)
    brainnet.train.utilities.load_checkpoint_from_setup(to_load, setup)

    print("Setup completed.")
    print(setup)

    return pred_step, dataloaders


def predict(args):
    """

    args = parse_args("blabla topofit t1w_1mm 780 validation".split())

    """

    setup = get_setup_at_checkpoint(args.specs, args.model, args.checkpoint)
    pred_step, dataloaders = create_predictor(args.model, setup, args.subset)

    out_dir = (
        setup.results.evaluation_dir / args.subset / f"checkpoint-{args.checkpoint:05d}"
    )
    print(f"Output dir    {out_dir}")

    if args.csv is None:
        content = list(
            itertools.chain(
                *[
                    itertools.product([k], map(str, v.dataset.subjects))
                    for k, v in dataloaders.items()
                ]
            )
        )
    else:
        with open(args.csv, "r") as f:
            content = csv.reader(f)

    subid2index = {
        k: dict(zip(map(str, v.dataset.subjects), range(len(v.dataset))))
        for k, v in dataloaders.items()
    }
    for dataset, subject in tqdm.tqdm(content):
        out_dir_sub = out_dir / dataset / subject
        if not out_dir_sub.exists():
            out_dir_sub.mkdir(parents=True)

        ds = dataloaders[dataset].dataset
        i = subid2index[dataset][subject]

        images, vox2ras, surfaces, template = ds[i]
        # image = images["t1w"]
        # template = template
        # vox2ras = vox2ras["t1w"]

        batch = ds[i]
        y_pred = pred_step(None, batch)
        y_pred = pred_step.postprocess(y_pred)

        train.write_example_prediction(y_pred, out_dir_sub)

        write_surfaces(
            y_pred["surface"],
            out_dir_sub,
            FREESURFER_VOLUME_INFO | dict(volume=image.shape[-3:]),
        )

    # for dataloader in dataloaders.values():
    #     dataset = dataloader.dataset
    #     print(dataset)
    #     first_img = dataset.images[0]
    #     img = getattr(IMAGE.images, first_img)
    #     ds_dir = dataset.ds_dir / dataset.name
    #     for i, sub in enumerate(dataset.subjects):
    #         # print(f"subject {i+1:4d} of {len(dataset):4d}")
    #         affine = torch.tensor(
    #             nib.load(ds_dir / sub / img.filename).affine, dtype=torch.float
    #         )
    #         image, y_pred, adjusted_affine, y_true = pred_step(None, dataset[i], affine)
    #         y_true = dict(surface=pred_step.get_surfaces(y_true))
    #         loss = pred_step.compute_loss(y_pred, y_true)
    #         # if loss["raw"]["white"]["chamfer"] > 0.4:
    #         print(f"{i:04d} : {sub:10s} : {loss['raw']['white']['chamfer']:10.4f}")

    #         this_out = out_dir / dataset.name / sub

    #         match args.format:
    #             case "torch":
    #                 # surfaces are in the voxel space of the cropped image
    #                 write_surface_torch(
    #                     y_pred["surface"],
    #                     adjusted_affine[0],  # @ torch.linalg.inv(affine),
    #                     this_out,
    #                     resolution=dataset.target_surface_resolution,
    #                 )
    #             case "freesurfer":
    #                 vol_info["volume"] = list(image.shape[-3:])
    #                 write_surface_freesurfer(
    #                     y_pred["surface"],
    #                     this_out,
    #                     adjusted_affine[0].numpy(),
    #                     vol_info,
    #                 )


# def write_surface_freesurfer(surfaces, out_dir, affine, volume_info):
#     if not out_dir.exists():
#         out_dir.mkdir(parents=True)
#     for hemi, s in surfaces.items():
#         for surf, ss in s.items():
#             # v = ss.vertices
#             v = ss.vertices
#             assert len(v) == 1
#             v = v[0]
#             v = v.cpu().numpy() @ affine[:3, :3].T + affine[:3, 3]
#             faces = ss.faces.cpu().numpy()
#             nib.freesurfer.write_geometry(
#                 out_dir / f"{hemi}.{surf}", v, faces, volume_info=volume_info
#             )

#             if "sigma" in ss.vertex_data:
#                 for v in ss.vertex_data["sigma"]:
#                     nib.freesurfer.write_morph_data(
#                         out_dir / f"{hemi}.{surf}.sigma",
#                         v.norm(dim=-1).detach().to(torch.float).cpu().numpy(),
#                     )


# def write_surface_torch(
#     surfaces: dict,
#     affine: torch.Tensor,
#     out_dir: Path,
#     prefix=None,
#     resolution=6,
#     label="prediction",
#     ext="pt",
# ):
#     if not out_dir.exists():
#         out_dir.mkdir(parents=True)
#     resolution = str(resolution)
#     for hemi, s in surfaces.items():
#         for surf, ss in s.items():
#             assert len(ss) == 1
#             merge = [hemi, surf, resolution, label, ext]
#             name = ".".join(merge if prefix is None else [prefix] + merge)
#             v = ss[0].cpu()
#             v = v @ affine[:3, :3].T + affine[:3, 3]
#             torch.save(v, out_dir / name)


def parse_args(argv):
    description = "Main interface to evaluating a BrainNet model."
    parser = argparse.ArgumentParser(
        prog="BrainNetEvaluator",
        description=description,
    )
    help_model = "The model to evaluate (e.g., topofit)."
    help_specs = "Configuration file defining the parameters for training."
    help_checkpoint = "Evaluate the model at checkpoint."
    help_subset = "Subset of data to evaluate on (e.g., train, validation, test)."

    parser.add_argument("model", help=help_model)
    parser.add_argument("specs", help=help_specs)
    parser.add_argument("checkpoint", type=int, help=help_checkpoint)
    parser.add_argument("subset", type=str, help=help_subset)
    parser.add_argument("--csv", default=None, type=str, help="")
    parser.add_argument(
        "--format",
        choices=["torch", "freesurfer"],
        default="freesurfer",
        help="Format in which to save predictions.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        nargs="+",
        help="Subset of data to evaluate on (e.g., train, validation, test).",
    )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    predict(args)
