#!/bin/bash

#SBATCH --job-name=spherereg         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=2           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=8G                   # Job memory request
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

module load freesurfer/8.0.0

source ~/mambaforge/etc/profile.d/conda.sh
conda activate simnibs-dev

RUN=t1w-1mm-reg
CKPT=best
# e.g., ADNI-GO2
DATASET=""
sub=$(find $SCRATCH/results/TopoFit/$RUN/evaluation/validation/checkpoint-$CKPT/$DATASET -type d -name "*sub-*" | sed -n "${SLURM_ARRAY_TASK_ID}p")

cd $sub
# cd sphere_pred

echo Running FreeSurfer commands in $PWD

# ?h.white is the predicted surface.
for hemi in lh rh
do
    # steps taken from recon-all dev table
    mris_smooth -n 3 -nw $hemi.white $hemi.smoothwm
    mris_inflate $hemi.smoothwm $hemi.inflated

    mris_curvature -w $hemi.white
    mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 $hemi.inflated

    mris_sphere -threads 2 $hemi.inflated $hemi.sphere
    CURV_ATLAS=$FREESURFER_HOME/average/$hemi.average.curvature.filled.buckner40.tif
    # CURV_ATLAS=$FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif
    mris_register -curv $hemi.sphere $CURV_ATLAS $hemi.sphere.reg
    # clean
    # rm $hemi.curv $hemi.smoothwm $hemi.inflated $hemi.sphere

    FSAVG_WHITE=/home/jesperdn/repositories/simnibs-gitlab/simnibs/resources/templates/fsaverage_surf/${hemi}.white.gii
    FSAVG_SPHERE=/home/jesperdn/repositories/simnibs-gitlab/simnibs/resources/templates/fsaverage_surf/${hemi}.sphere.gii

    echo CAT: $hemi.sphere
    /home/jesperdn/repositories/simnibs-gitlab/simnibs/external/bin/linux/CAT_Surf2Sphere ${hemi}.white ${hemi}.sphere.cat.gii 10

    echo CAT: $hemi.sphere.reg
    /home/jesperdn/repositories/simnibs-gitlab/simnibs/external/bin/linux/CAT_WarpSurf -steps 2 -avg -i ${hemi}.white -is ${hemi}.sphere.cat.gii -t $FSAVG_WHITE -ts $FSAVG_SPHERE -ws ${hemi}.sphere.reg.cat.gii

    echo JOSA
    josareg --surfdir . -threads 2 --hemi $hemi --o . --no-link

    python -c "import cortech; import nibabel as nib; fsa = cortech.Sphere.from_file('/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/surf/$hemi.sphere.reg'); a,b,c=nib.freesurfer.read_annot('/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/label/$hemi.aparc.annot'); sub=cortech.Sphere.from_file('$hemi.sphere.reg'); nv=sub.n_vertices; x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.fs.annot', x,b,c); sub=cortech.Sphere.from_file('$hemi.josa.sphere.reg'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.josa.annot', x,b,c); sub=cortech.Sphere.from_file('$hemi.registration'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.pred.annot', x,b,c); sub=cortech.Sphere.from_file('/home/jesperdn/repositories/brainsynth/brainsynth/resources/$hemi.sphere.reg'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.map.annot', x[:nv],b,c); sub=cortech.Sphere.from_file('$hemi.sphere.reg.cat.gii'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.cat.annot', x,b,c);"
done

# python -c "import cortech; import nibabel as nib; fsa = cortech.Sphere.from_file('/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/surf/$hemi.sphere.reg'); a,b,c=nib.freesurfer.read_annot('/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/label/$hemi.aparc.annot'); sub=cortech.Sphere.from_file('$hemi.sphere.reg'); nv=sub.n_vertices; x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.fs.annot', x,b,c)"
