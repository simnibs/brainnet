from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

global CHECKPOINTS
global RUNS
global SUBSET
global METRIC

RUNS = ["t1w-1mm", "t1w-1mm-noUC"]
RUNS = ["synth-random", "synth-random-clinical", "synth-random-clinical-axial"]
RUNS = ["t1w-1mm-reg", "synth-1mm-reg", "synth-random-reg"]
RUNS = ["synth-random-exvivo", "synth-random-exvivo-mix"]
RUNS = ["synth-random-exvivo"]

CHECKPOINTS = [600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800]

# CHECKPOINTS = {"synth-random": [780],"synth-random-clinical": [760],"synth-random-clinical-axial": [760]}

SUBSET = "validation"
METRIC = "chamfer"

RESULTS_DIR = Path("/mnt/scratch/personal/jesperdn/results")
MODEL = "TopoFit"


def load_dataframes(run, checkpoints=None):
    eval_dir = RESULTS_DIR / MODEL / run / "evaluation"

    checkpoints = checkpoints or CHECKPOINTS

    # load multiple evluations
    dfs = {}
    for ckpt in checkpoints:
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


def write_best(run, idx):
    ckpt_dir = RESULTS_DIR / MODEL / run / "checkpoint"
    src_ckpt = ckpt_dir / f"state_checkpoint_{idx:05d}.pt"
    dest_ckpt = ckpt_dir / "state_checkpoint_best.pt"

    print(f"Run: {run}")
    print(f"Copying {src_ckpt.name} -> {dest_ckpt.name}")

    ckpt = torch.load(src_ckpt)
    state_dict = ckpt["model"]
    torch.save(state_dict, dest_ckpt)


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


# runs = {run: load_dataframes(run, CHECKPOINTS[run]) for run in RUNS}
runs = {run: load_dataframes(run) for run in RUNS}

fig = plot(runs)


fig = plot({"t1w-1mm-reg": runs["t1w-1mm-reg"]})
fig = plot({"synth-1mm-reg": runs["synth-1mm-reg"]})
fig = plot({"synth-random-reg": runs["synth-random-reg"]})

best_idx = {}
for k, v in runs.items():
    print(f"Best checkpoint for {k}")
    print()
    best_idx[k] = find_best(v)
    print("\n")

for k, v in best_idx.items():
    write_best(k, v["white"])


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
