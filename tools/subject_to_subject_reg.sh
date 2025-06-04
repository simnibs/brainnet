#!/bin/bash

#SBATCH --job-name=inflate         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=1-489           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

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
conda activate simnibs-dev

model="t1w-1mm"
checkpoint="00780"
ROOT_PRED=$SCRATCH/results/TopoFit/$model/evaluation/validation/checkpoint-$checkpoint
ROOT_TRUE=$CORTECH/nobackup/training_data/full

SUB_DIR=$(find $ROOT_PRED -maxdepth 2 -type d -name "*sub-*" | sed -n "${SLURM_ARRAY_TASK_ID}p")
subid=$(basename $SUB_DIR)
dataset=$(basename $(dirname $SUB_DIR))

cd $SUB_DIR

echo Running FreeSurfer commands in $PWD

mkdir ./TRUE
mkdir ./PRED

for hemi in lh #rh
do
    CURV_ATLAS=$FREESURFER_HOME/average/$hemi.average.curvature.filled.buckner40.tif
    FS_APARC=$FREESURFER_HOME/subjects/fsaverage/label/$hemi.aparc.annot
    FS_SURF=$FREESURFER_HOME/subjects/fsaverage/surf/$hemi.sphere.reg

    # TRUE
    # =====================
    cd TRUE
    cp $ROOT_TRUE/$dataset/$subid/$hemi.white .

    # steps taken from recon-all dev table
    mris_smooth -n 3 -nw $hemi.white $hemi.smoothwm
    mris_inflate $hemi.smoothwm $hemi.inflated
    mris_curvature -w $hemi.white
    mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 $hemi.inflated
    mris_sphere $hemi.inflated $hemi.sphere

    cd ..

    # PRED
    # =====================
    cd PRED

    cp ../$hemi.white .

    # steps taken from recon-all dev table
    mris_smooth -n 3 -nw $hemi.white $hemi.smoothwm
    mris_inflate $hemi.smoothwm $hemi.inflated
    mris_curvature -w $hemi.white
    mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 $hemi.inflated
    mris_sphere $hemi.inflated $hemi.sphere

    # register to FSAVERAGE
    mris_register -curv $hemi.sphere $CURV_ATLAS $hemi.sphere.reg

    cd ..

    # register TRUE to PRED
    # =====================
    mris_register -1 -curv ./TRUE/$hemi.sphere ./PRED/$hemi.sphere ./TRUE/$hemi.sphere.reg2pred

    # resample PRED on TRUE
    python -c "import cortech; import nibabel as nib; pred = cortech.SphericalRegistration.from_freesurfer('./PRED/$hemi.sphere'); true = cortech.SphericalRegistration.from_freesurfer('./TRUE/$hemi.sphere.reg2pred'); true_surf = cortech.Surface.from_freesurfer('./TRUE/$hemi.white'); pred.faces = pred.faces.astype(int); true.faces = true.faces.astype(int); true.project(pred); v = true.resample(true_surf.vertices); nib.freesurfer.write_geometry('./PRED/$hemi.white.resampled_on_true', v, pred.faces);"

    # resample APARC on PRED
    python -c "import nibabel as nib; import cortech; label, ctab, names = nib.freesurfer.read_annot('$FS_APARC'); fsavg = cortech.SphericalRegistration.from_freesurfer('$FS_SURF'); subject = cortech.SphericalRegistration.from_freesurfer('./PRED/$hemi.sphere.reg'); fsavg.project(subject, method='nearest'); subject_label = fsavg.resample(label); nib.freesurfer.write_annot('./PRED/${hemi}.aparc.annot', subject_label, ctab, names);"

done
