from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


subset = "validation"
out_dir = Path(
    "/mnt/scratch/personal/jesperdn/results/TopoFit/01_lh_t1w-adapt-sourceonlyN1_1mm/evaluation"
)
metric = "loss.pickle"
eval_dir = out_dir / subset

# df = pd.read_pickle(eval_dir / metric)

# load multiple evluations

dfs = {}
for i in sorted(eval_dir.glob("loss-*.pickle")):
    name = i.stem.lstrip("loss-")
    dfs[name] = pd.read_pickle(i)

# join datasets on index (subjects)
df = pd.concat(dfs, names = ["dataset", "subject"])

df = dfa-dfsynth

names = df.index.unique(level="dataset")
data = [df.loc[k]["white", "chamfer"] for k in names]

plt.figure()
_ = plt.boxplot(data,labels=names)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.ylim([-0.3, 0.3])

