#!/bin/bash

#SBATCH --job-name=resample2fsaverage         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=1-81           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

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

module load freesurfer/7.4.1

source ~/mambaforge/etc/profile.d/conda.sh
conda activate simnibs-dev

model="t1w-1mm"
checkpoint="00780"
ROOT_PRED=$SCRATCH/results/TopoFit/$model/evaluation/validation/checkpoint-$checkpoint
ROOT_TRUE=$CORTECH/nobackup/training_data/full

SUB_DIR=$(find $ROOT_PRED -maxdepth 2 -type d -name "*sub-*" | sed -n "${SLURM_ARRAY_TASK_ID}p")
subid=$(basename $SUB_DIR)
dataset=$(basename $(dirname $SUB_DIR))

echo $SUB_DIR
cd $SUB_DIR

for hemi in lh # rh
do
    FS_SURF=$FREESURFER_HOME/subjects/fsaverage/surf/$hemi.sphere.reg

    python -c "import nibabel as nib; import cortech; fsavg = cortech.SphericalRegistration.from_freesurfer('$FS_SURF'); subject = cortech.SphericalRegistration.from_freesurfer('./PRED/$hemi.sphere.reg'); fsavg.faces = fsavg.faces.astype(int); subject.faces = subject.faces.astype(int); subject.project(fsavg); sigma = nib.freesurfer.read_morph_data('./$hemi.white.sigma'); sigma_fs = subject.resample(sigma); nib.freesurfer.write_morph_data('./${hemi}.white.sigma.fsaverage', sigma_fs); sigma = nib.freesurfer.read_morph_data('./$hemi.pial.sigma'); sigma_fs = subject.resample(sigma); nib.freesurfer.write_morph_data('./${hemi}.pial.sigma.fsaverage', sigma_fs);distance = nib.freesurfer.read_morph_data('./$hemi.white.distance'); distance_fs = subject.resample(distance); nib.freesurfer.write_morph_data('./${hemi}.white.distance.fsaverage', distance_fs); distance = nib.freesurfer.read_morph_data('./$hemi.pial.distance'); distance_fs = subject.resample(distance); nib.freesurfer.write_morph_data('./${hemi}.pial.distance.fsaverage', distance_fs);"

done