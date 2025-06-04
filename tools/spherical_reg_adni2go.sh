#!/bin/bash

#SBATCH --job-name=sph-reg         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=1-200           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

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
#source ~/mambaforge/etc/profile.d/conda.sh
#conda activate synth

ROOT_SOURCE=$CORTECH/nobackup/jesper/evaluation_data/ADNI-GO2
ROOT_DEST=$SCRATCH/ADNI-GO2

# find /mnt/scratch/personal/jesperdn/training_data/spherereg/ -type d -name "*sub-*" > allsubs.txt
SRC_SUB_DIR=$(find $ROOT_SOURCE -maxdepth 2 -type d -name "sub-*" | sed -n "${SLURM_ARRAY_TASK_ID}p")
DST_SUB_DIR=${SRC_SUB_DIR/${ROOT_SOURCE}/${ROOT_DEST}}

echo $SRC_SUB_DIR
echo $DST_SUB_DIR
mkdir -p $DST_SUB_DIR
cd $DST_SUB_DIR
echo Running FreeSurfer commands in $PWD

# ?h.white is the predicted surface.
for hemi in lh rh
do
    # CURV_ATLAS=$FREESURFER_HOME/average/$hemi.average.curvature.filled.buckner40.tif
    CURV_ATLAS=$FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif

    # steps taken from recon-all dev table
    mris_smooth -n 3 -nw $SRC_SUB_DIR/$hemi.white $hemi.smoothwm
    mris_inflate $hemi.smoothwm $hemi.inflated
    # mris_curvature -w ?h.white
    # mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 $hemi.inflated
    mris_sphere $hemi.inflated $hemi.sphere
    mris_register -curv $hemi.sphere $CURV_ATLAS $hemi.sphere.reg
    # clean
    # rm $hemi.curv $hemi.smoothwm $hemi.inflated $hemi.sphere
done
