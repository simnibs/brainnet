#!/bin/bash

#SBATCH --job-name=inflate         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=1-4753           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

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

# find /mnt/scratch/personal/jesperdn/training_data/spherereg/ -type d -name "*sub-*" > allsubs.txt
sub=$(cat /mnt/scratch/personal/jesperdn/training_data/allsubs.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")

BRAINSYNTH_TOOLS_DIR=/mrhome/jesperdn/repositories/brainsynth/tools

cd $sub
echo Running FreeSurfer commands in $PWD

# ?h.white is the predicted surface.
for hemi in lh rh
do
    # steps taken from recon-all dev table
    mris_smooth -n 3 -nw $hemi.white $hemi.smoothwm
    mris_inflate $hemi.smoothwm $hemi.inflated
    mris_sphere $hemi.inflated $hemi.sphere
    CURV_ATLAS=$FREESURFER_HOME/average/$hemi.average.curvature.filled.buckner40.tif
    mris_register -curv $hemi.sphere $CURV_ATLAS $hemi.sphere.reg
    # clean
    # rm $hemi.curv $hemi.smoothwm $hemi.inflated $hemi.sphere
done
