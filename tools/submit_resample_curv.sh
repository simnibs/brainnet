#!/bin/bash

#SBATCH --job-name=spherereg         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=4G                   # Job memory request
#SBATCH --array=1-286           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

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

module load freesurfer/8.0.0

source ~/mambaforge/etc/profile.d/conda.sh
conda activate simnibs-dev

# find /mnt/scratch/personal/jesperdn/training_data/spherereg/ -type d -name "*sub-*" > allsubs.txt
# sub=$(cat /mnt/scratch/personal/jesperdn/training_data/allsubs.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")

# sub=$(cat /mnt/scratch/personal/jesperdn/training_data/allsubs.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
# RUN=t1w-1mm-registration-reg001
RUN=t1w-1mm-reg
CKPT=00680
sub=$(find $SCRATCH/results/TopoFit/$RUN/evaluation/validation/checkpoint-$CKPT -type d -name "*sub-*" | sed -n "${SLURM_ARRAY_TASK_ID}p")

cd $sub
echo Running FreeSurfer commands in $PWD

# ?h.white is the predicted surface.
for hemi in lh rh
do
    python -c "import cortech; import nibabel as nib; fsa = cortech.Sphere.from_file('/mnt/depot64/freesurfer/freesurfer.8.0.0/subjects/fsaverage/surf/$hemi.sphere'); fsa.faces=fsa.faces.astype(int); curv=nib.freesurfer.read_morph_data('$hemi.white.H'); sub=cortech.Sphere.from_file('$hemi.sphere.reg'); sub.faces=sub.faces.astype(int); x = sub.project_and_resample(fsa, curv); nib.freesurfer.write_morph_data('$hemi.white.H.fs', x); sub=cortech.Sphere.from_file('$hemi.sphere.reg.cat.gii'); sub.faces=sub.faces.astype(int); x = sub.project_and_resample(fsa, curv); nib.freesurfer.write_morph_data('$hemi.white.H.cat', x); sub=cortech.Sphere.from_file('$hemi.josa.sphere.reg'); sub.faces=sub.faces.astype(int); x = sub.project_and_resample(fsa, curv); nib.freesurfer.write_morph_data('$hemi.white.H.josa', x); sub=cortech.Sphere.from_file('$hemi.registration'); sub.faces=sub.faces.astype(int); x = sub.project_and_resample(fsa, curv); nib.freesurfer.write_morph_data('$hemi.white.H.pred', x); sub=cortech.Sphere.from_file('$hemi.josa.sphere.reg'); sub.faces=sub.faces.astype(int); x = sub.project_and_resample(fsa, curv); nib.freesurfer.write_morph_data('$hemi.white.H.josa', x); sub=cortech.Sphere.from_file('/home/jesperdn/repositories/brainsynth/brainsynth/resources/$hemi.sphere.reg'); sub.faces=sub.faces.astype(int); x = sub.project_and_resample(fsa, curv); nib.freesurfer.write_morph_data('$hemi.white.H.map', x)"
done
