#!/bin/bash

#SBATCH --job-name=brainnet         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=6          # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1

#SBATCH --exclusive

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

# By default, functions are not exported to be available in subshells so we
# need this before we can use 'conda activate'
source ~/mambaforge/etc/profile.d/conda.sh
conda activate synth

repo=/home/jesperdn/repositories/brainnet

find $repo -name __pycache__ -exec rm -r {} +

SCRIPT="${repo}/brainnet/train/brainnet_train.py"
CONFIG="brainnet.config.topofit.mri.main"
# ARGS="--max-epochs 800"
ARGS="--load-checkpoint 800 --max-epochs 1400"

cmd="python $SCRIPT $CONFIG $ARGS"

echo
echo "Executing : $cmd"
echo

$cmd