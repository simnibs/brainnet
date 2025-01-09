from pathlib import Path
import nibabel as nib
import numpy as np
# import pandas as pd

from cortech.surface import Surface


import sys

if __name__ == "__main__":

    model, index = sys.argv[1:]
    index = int(index)

    print(f"Model: {model}")
    # model = "topofit-ours"
    # model = "topofit-ds"
    # model = "v2c-flow"

    root_dir = Path("/mnt/scratch/personal/jesperdn/")
    datadir = Path("/mnt/projects/CORTECH/nobackup/training_data/full")

    assert model in {"topofit-ds", "topofit-ours", "v2c-flow"}

    subdir = {"topofit-ds": "topofit", "topofit-ours": "topofit-ours", "v2c-flow": "vox2cortex"}

    d = root_dir / subdir[model] / "t2w" / "validation"

    print(d)

    valid_topofit = np.where(np.load("/home/jesperdn/repositories/brainnet/medial_wall.npz")["arr_0"] == 0)[0]
    valid_topofit = dict(lh=valid_topofit, rh=valid_topofit)
    valid_v2c = dict(
        lh=np.sort(nib.freesurfer.read_label("/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/label/lh.cortex.label")),
        rh=np.sort(nib.freesurfer.read_label("/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/label/rh.cortex.label")),
    )
    valid = {"topofit-ds": valid_topofit, "topofit-ours": valid_topofit, "v2c-flow": valid_v2c}


    filenames = {"topofit-ds": {
        ("lh", "white"): "cortex-int-lh.srf",
        ("rh", "white"): "cortex-int-rh.srf",
        ("lh", "pial"): "cortex-ext-lh.srf",
        ("rh", "pial"): "cortex-ext-rh.srf",
        },
        "topofit-ours": {
        ("lh", "white"): "lh.white.pred",
        ("rh", "white"): "rh.white.pred",
        ("lh", "pial"): "lh.pial.pred",
        ("rh", "pial"): "rh.pial.pred",
        },
        "v2c-flow": {
        ("lh", "white"): "lh.white.pred",
        ("rh", "white"): "rh.white.pred",
        ("lh", "pial"): "lh.pial.pred",
        ("rh", "pial"): "rh.pial.pred",
        },
    }

    subjects = [(k.stem, i.stem) for k in sorted(d.glob("*")) for i in sorted(k.glob("sub-*"))]

    # hemi = ("lh", "rh")
    # surfaces = ("white", "pial")
    # metric = "Distace"

    # row_index = pd.MultiIndex.from_tuples(subjects)
    # col_index = pd.MultiIndex()
    # df = pd.DataFrame()

    valid_vertices = valid[model]

    ds,sub = subjects[index]

    print(f"Subject: {ds:10s} {sub:10s}")

    from scipy.spatial import cKDTree

    data = {}
    for surf in ("white", "pial"):
        for h in ("lh", "rh"):
            v_true, f_true, m_true = nib.freesurfer.read_geometry(datadir / ds / sub / f"{h}.{surf}", read_metadata=True)
            v_pred, f_pred, m_pred = nib.freesurfer.read_geometry(d / ds / sub / filenames[model][h,surf], read_metadata=True)
            affine = np.stack((m_true["xras"], m_true["yras"],m_true["zras"]),axis=1)
            affine2 = np.stack((m_pred["xras"], m_pred["yras"],m_pred["zras"]),axis=1)
            v_pred = v_pred @ affine @ np.linalg.inv(affine2) - m_true["cras"] + m_pred["cras"]

            s_pred = Surface(v_pred,f_pred)
            s_true = Surface(v_true,f_true)

            true_to_pred = s_pred.distance_query(s_true.vertices[valid_vertices[h]])
            pred_to_true = s_true.distance_query(s_pred.vertices[valid_vertices[h]])

            a,_ = cKDTree(s_pred.vertices).query(s_true.vertices[valid_vertices[h]])
            c,_ = cKDTree(s_true.vertices).query(s_pred.vertices[valid_vertices[h]])

            assert np.all(a >= true_to_pred), f"ERROR 0 {true_to_pred[a < true_to_pred]} {a[a < true_to_pred]}"
            assert np.all(c >= pred_to_true), f"ERROR 1 {pred_to_true[c < pred_to_true]} {c[c < pred_to_true]}"

            if np.isnan(true_to_pred).any():
                print("NANS")
                print(surf,h)
                print(np.isnan(true_to_pred).sum())
            if np.isnan(pred_to_true).any():
                print("NANS")
                print(surf,h)
                print(np.isnan(pred_to_true).sum())

            # true_to_pred1 = s_pred.distance_query(s_true.vertices[valid_vertices[h]])
            # pred_to_true1 = s_true.distance_query(s_pred.vertices[valid_vertices[h]])

            # assert np.all(a >= true_to_pred1), f"err 2 {true_to_pred1[a < true_to_pred1]} {a[a < true_to_pred1]}"
            # assert np.all(c >= pred_to_true1), f"err 3 {pred_to_true1[c < pred_to_true1]} {c[c < pred_to_true1]}"

            # if not np.allclose(true_to_pred, true_to_pred1):
            #     print("ALLCLOSE")
            #     print(surf, h)
            #     q = np.argmax(np.abs(true_to_pred-true_to_pred1))
            #     print(q)
            #     print(s_true.vertices[valid_vertices[h]][q]+ m_true["cras"])
            #     print(true_to_pred[q])
            #     print(true_to_pred1[q])
            # if not np.allclose(pred_to_true, pred_to_true1):
            #     print("ALLCLOSE")
            #     print(surf, h)
            #     q = np.argmax(np.abs(pred_to_true-pred_to_true1))
            #     print(q)
            #     print(s_pred.vertices[valid_vertices[h]][q])
            #     print(pred_to_true[q])
            #     print(pred_to_true1[q])

            data[f"{surf}_{h}_true-to-pred"] = true_to_pred
            data[f"{surf}_{h}_pred-to-true"] = pred_to_true

    np.savez(d / ds / sub / "distances.npz", **data)
    print(f"wrote {d / ds / sub / 'distances.npz'}")
    # np.savez("distances2.npz", **data)

    # data = {}
    # for surf in ("white", "pial"):
    #     for h in ("lh", "rh"):
    #         v, f = nib.freesurfer.read_geometry(d / ds / sub / filenames[model][h,surf])
    #         s = Surface(v,f)
    #         sif = np.unique(s.self_intersections().ravel()).size
    #         data[f"{surf}_{h}"] = sif / s.n_faces * 100

    # np.savez(d / ds / sub / "self_intersections.npz", **data)
