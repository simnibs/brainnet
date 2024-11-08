#!/bin/bash

#SBATCH --job-name=inflate         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=1-34           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

echo "Job Information"
echo
echo "Job name     :  $SLURM_JOB_NAME"
echo "Job ID       :  $SLURM_ARRAY_JOB_ID"
echo "Task ID      :  $SLURM_ARRAY_TASK_ID"
echo "Cluster name :  $SLURM_CLUSTER_NAME"
echo "Node name    :  $SLURM_NODENAME"
echo "Date         :  $(date)"
echo "Working dir  :  $SLURM_SUBMIT_DIR"
echo

module load freesurfer/7.4.0

source ~/mambaforge/etc/profile.d/conda.sh
conda activate synth

ds=Buckner40
root=/mnt/scratch/personal/jesperdn/training_data/spherereg/$ds
sub=$(ls -1 $root | sed -n "${SLURM_ARRAY_TASK_ID}p")

echo $root
echo $sub

# Left

hemi=lh

torchsurf=$root/$sub/${hemi}.white.6.prediction.pt
insurf=$root/$sub/freesurfer.${hemi}.white.6.prediction
inflated=$root/$sub/${hemi}.inflated
smoothwm=$root/$sub/${hemi}.smoothwm
sphere=$root/$sub/${hemi}.sphere
sulc=${hemi}.sulc

python /mrhome/jesperdn/repositories/brainsynth/tools/torch_vertices_to_freesurfer_surface.py $torchsurf $hemi

mris_inflate -sulc $sulc $insurf $inflated

# gets prepended with rh... No idea why. Probably because some header info is missing
mv $root/$sub/rh.$sulc $root/$sub/$sulc

ln -s $insurf  $smoothwm
mris_sphere $inflated $sphere

python -c "import nibabel as nib; import torch; v, _ = nib.freesurfer.read_geometry('$inflated'); torch.save(torch.tensor(v, dtype=torch.float), '${inflated}.pt'); v, _ = nib.freesurfer.read_geometry('$sphere'); torch.save(torch.tensor(v, dtype=torch.float), '${sphere}.pt')"

rm $insurf $inflated $smoothwm $sphere

# Right

hemi=rh

torchsurf=$root/$sub/${hemi}.white.6.prediction.pt
insurf=$root/$sub/freesurfer.${hemi}.white.6.prediction
inflated=$root/$sub/${hemi}.inflated
smoothwm=$root/$sub/${hemi}.smoothwm
sphere=$root/$sub/${hemi}.sphere
sulc=${hemi}.sulc

python /mrhome/jesperdn/repositories/brainsynth/tools/torch_vertices_to_freesurfer_surface.py $torchsurf $hemi

mris_inflate -sulc $sulc $insurf $inflated

# gets prepended with rh... No idea why. Probably because some header info is missing
mv $root/$sub/rh.$sulc $root/$sub/$sulc

ln -s $insurf  $smoothwm
mris_sphere $inflated $sphere

python -c "import nibabel as nib; import torch; v, _ = nib.freesurfer.read_geometry('$inflated'); torch.save(torch.tensor(v, dtype=torch.float), '${inflated}.pt'); v, _ = nib.freesurfer.read_geometry('$sphere'); torch.save(torch.tensor(v, dtype=torch.float), '${sphere}.pt')"

rm $insurf $inflated $smoothwm $sphere