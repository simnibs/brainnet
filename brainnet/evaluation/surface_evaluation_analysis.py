from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

root_dir = Path("/mnt/scratch/personal/jesperdn/")

subdir = {"topofit-ds": "topofit", "topofit-ours": "topofit-ours", "v2c-flow": "vox2cortex"}


def assd(a,b):
    return float(np.concatenate((a,b)).mean())

def hausdorff(a,b, p=90):
    return float(max(np.percentile(a, p), np.percentile(b, p)))

metrics = dict(assd=assd, hd90=hausdorff)

subset = "train"
contrast = "synth_1mm"

model = "topofit-ours"
d = root_dir / subdir[model]  / contrast / subset
subjects = [(k.stem, i.stem) for k in sorted(d.glob("*")) for i in sorted(k.glob("sub-*"))]
datasets = np.unique([i[0] for i in subjects]).tolist()


models = ["topofit-ds", "topofit-ours", "v2c-flow"]
models = ["topofit-ours"]

subs = {}
results = {}

for model in models:
    print(model)
    for ds,sub in tqdm(subjects):
        d = root_dir / subdir[model] / contrast / subset
        data = np.load(d / ds / sub / "distances.npz")

        for m in metrics:
            for s in ("white", "pial"):
                k = (m,s,ds,model)
                if k not in results:
                    results[k] = []
                    subs[k] = []

                a1 = data[f"{s}_lh_true-to-pred"]
                a2 = data[f"{s}_lh_pred-to-true"]
                b1 = data[f"{s}_rh_true-to-pred"]
                b2 = data[f"{s}_rh_pred-to-true"]

                if np.isnan(a1).any() or np.isnan(a2).any() or np.isnan(b1).any() or np.isnan(b2).any():
                    print(f"{ds:15s} : {sub:10s}")

                a1 = a1[~np.isnan(a1)]
                a2 = a2[~np.isnan(a2)]
                b1 = b1[~np.isnan(b1)]
                b2 = b2[~np.isnan(b2)]

                results[k].append((metrics[m](a1,a2) + metrics[m](b1,b2)) / 2.0)
                subs[k].append(sub)



print(f"{'dataset':15s} {models[0]:10s} {models[1]:10s} {models[2]:10s}")
for ds in datasets:
    res = [np.mean(results["hd90", "pial", ds, model]) for model in models]
    print(f'{ds:15s} {res[0]:10.3f} {res[1]:10.3f} {res[2]:10.3f}')


N = 20
x = np.concatenate([results["assd", "white", ds, "topofit-ours"] for ds in datasets])
s = np.argsort(x)[::-1]
y = [subjects[i] for i in s]
xs = x[s]
for i in range(N):
    print(f"{y[i][0]:10s} {y[i][1]:10s} : {xs[i]:10.3f}")


N = 20
xx = np.concatenate([results["assd", "pial", ds, "topofit-ours"] for ds in datasets])
s = np.argsort(xx)[::-1]
yy = [subjects[i] for i in s]
xxs = xx[s]


fig, axes = plt.subplots(1,1)
axes.plot(xs)
axes.plot(xxs)
axes.legend(["white", "pial"])
axes.set_title("ASSD")
fig.show()


fig, axes = plt.subplots(1,1)
axes.plot(xs)
axes.plot(xxs)
axes.legend(["white", "pial"])
axes.set_title("HD90")
fig.show()


for ds,sub in y[99:102]:

for ds,sub in y[13:N]:
    print(f"{ds:10s} {sub:10s}")

    ds=y[1][0]# "ADNI3"
    sub=y[1][1] # "sub-152"
    surf="white"
    contrast="synth_1mm"
    subset="train"

    cmd = f"vglrun freeview \
        /mnt/projects/CORTECH/nobackup/training_data/full/{ds}/{sub}/T1w.nii \
        -f /mnt/scratch/personal/jesperdn/topofit-ours/{contrast}/{subset}/{ds}/{sub}/lh.{surf}.pred \
        /mnt/scratch/personal/jesperdn/topofit-ours/{contrast}/{subset}/{ds}/{sub}/rh.{surf}.pred \
        --edgecolor blue \
        -f /mnt/scratch/personal/jesperdn/topofit-ours/t1w/{subset}/{ds}/{sub}/lh.{surf}.pred \
        /mnt/scratch/personal/jesperdn/topofit-ours/t1w/{subset}/{ds}/{sub}/rh.{surf}.pred \
        --edgecolor green \
        -f /mnt/scratch/personal/jesperdn/topofit/t1w/{subset}/{ds}/{sub}/cortex-int-lh.srf  \
        /mnt/scratch/personal/jesperdn/topofit/t1w/{subset}/{ds}/{sub}/cortex-int-rh.srf \
        --edgecolor red \
        -f /mnt/projects/CORTECH/nobackup/training_data/full/{ds}/{sub}/lh.{surf} \
        /mnt/projects/CORTECH/nobackup/training_data/full/{ds}/{sub}/rh.{surf}"



    # freeview \
    #     /mnt/projects/CORTECH/nobackup/training_data/full/${ds}/${sub}/T1w.nii \
    #     -f /mnt/scratch/personal/jesperdn/topofit-ours/$contrast/$subset/${ds}/${sub}/*h.${surf}.pred \
    #     --edgecolor red -f /mnt/projects/CORTECH/nobackup/training_data/full/${ds}/${sub}/*h.${surf}

    res = subprocess.run(cmd.split(), check=True)



[subs["assd", "white", ds, "topofit-ours"][np.argmax(results["assd", "white", ds, "topofit-ours"])] for ds in datasets]


results = {}
subs = {}
for s in ("white", "pial"):
    for model in models:
        for ds,sub in subjects:
            d = root_dir / subdir[model]
            k = (s,ds,model)
            if k not in results:
                results[k] = []
                subs[k] = []

            data = np.load(d / ds / sub / "self_intersections.npz")
            results[k].append(float((data[f"{s}_lh"] + data[f"{s}_rh"])) / 2.0)

            subs[k].append(sub)
            # if np.isnan(results[k][-1]):
            #     raise ValueError

print(f"{'dataset':15s} {models[0]:10s} {models[1]:10s} {models[2]:10s}")
for ds in datasets:
    res = [np.mean(results["pial", ds, model]) for model in models]
    print(f'{ds:15s} {res[0]:10.4f} {res[1]:10.4f} {res[2]:10.4f}')


import pandas as pd

df = pd.DataFrame.from_dict()