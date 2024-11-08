from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

out_dir = Path("/home/jesperdn/nobackup/brainreg_slices/")
data_dir = Path("/mnt/projects/CORTECH/nobackup/training_data/brainreg")

# save slices
for i in tqdm(sorted(data_dir.glob("*T1w.areg-mni.nii"))):
    parts = i.stem.split(".")
    img = nib.load(i)
    s = img.dataobj[96]
    plt.imsave(out_dir / f"{parts[0]}.{parts[1]}.png", s.T, vmin=0, vmax=255, origin="lower")

# view (bash)
"""

for i in *.png; do
    eog --single-window $i
    sleep 0.2
done;

"""