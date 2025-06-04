from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


global CHECKPOINTS
global RUNS
global SUBSET
global METRIC

RUNS = ["t1w-1mm", "t1w-1mm-noUC"]
RUNS = ["synth-random", "synth-random-clinical"]

RUNS = ["synth-random"]
SUBSET = "validation"
CHECKPOINTS = [600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800]
CHECKPOINTS = [800]
METRIC = "chamfer"


def load_dataframes(run):
    eval_dir = Path(f"/mnt/scratch/personal/jesperdn/results/TopoFit/{run}/evaluation")

    # load multiple evluations
    dfs = {}
    for ckpt in CHECKPOINTS:
        dfs[ckpt] = pd.read_pickle(eval_dir / f"{SUBSET}-checkpoint-{ckpt:05d}.pickle")
        dfs[ckpt].index = dfs[ckpt].index.set_names(["dataset", "subject"])
    return pd.concat(dfs, names=["checkpoint"])


def find_best(df):
    mean = df.groupby("checkpoint").mean()
    idx = mean.idxmin()

    print("Surface    Metric       Checkpoint   Value")
    print("--------------------------------------------")
    for (s, m), ckpt in zip(idx.index, idx):
        print(f"{s:10s} {m:10s} {ckpt:5d}       {mean.loc[ckpt, (s, m)]:10.5f}")

    print()

    print(f"Best checkpoint based on metric '{METRIC}'")
    print(idx[:, METRIC])
    return idx[:, METRIC]


def plot(run_dict, metric=None):
    if metric is None:
        metric = METRIC
    fig, axes = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    for s, ax in zip(("white", "pial"), axes):
        for k, df in run_dict.items():
            mean = df[s, metric].groupby("checkpoint").mean()
            res = mean.idxmin()
            # label = k if s == "pial" else None
            _ = ax.plot(mean, label=k)
            ax.scatter(res, mean.loc[res], c="red")
        ax.set_title(s)
        ax.grid(alpha=0.25)
    axes[-1].legend()
    return fig


runs = {run: load_dataframes(run) for run in RUNS}

fig = plot(runs)

for k, v in runs.items():
    print(f"Best checkpoint for {k}")
    print()
    idx = find_best(v)
    print("\n")


for k, v in runs.items():
    idx = best[k]
    print(k)
    values = v.loc[idx, pd.IndexSlice[HEMI, AFFINE, METRIC]].sort_values(
        ascending=False
    )
    print(values[:10])
    subs = values.index[:10]
    subs = list(zip(subs.get_level_values("dataset"), subs.get_level_values("subject")))
    filename = f"topofit_subjects_{k}.csv"

    with open(filename, "w") as f:
        csv_writer = csv.writer(f)
        for row in subs:
            csv_writer.writerow(row)


names = df.index.unique(level="dataset")
data = [df.loc[k]["white", "chamfer"] for k in names]


df.loc[:, pd.IndexSlice[:, "MSE"]]

df.loc[:, pd.IndexSlice["lh_brain"]]

data = [df.loc[k]["lh_brain", "MSE"] for k in names]


plt.figure()
_ = plt.boxplot(data, tick_labels=names)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
# plt.ylim([-0.3, 0.3])
