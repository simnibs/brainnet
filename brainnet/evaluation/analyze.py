from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


metric = "loss.pickle"
out_dir = Path("/mnt/scratch/personal/jesperdn/results/Cortex/12_lh_T1w_1mm/evaluation")
subset = "validation"

df = pd.read_pickle(out_dir / subset / metric)

