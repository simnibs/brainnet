#!/bin/bash

#SBATCH --job-name=spherereg         # Job name
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=2           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --array=0-1           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

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

i=00440
EXAM_DIR=$SCRATCH/results/TopoFit/synth-random-reg/examples
DO_CAT=true

HEMISPHERES=("lh" "rh")
hemi=${HEMISPHERES[$SLURM_ARRAY_TASK_ID]}

cd $EXAM_DIR

# for hemi in lh rh
# do
cp epoch-$i.validation.pred.$hemi.white $hemi.white

# steps taken from recon-all dev table
mris_smooth -n 3 -nw $hemi.white $hemi.smoothwm
mris_inflate $hemi.smoothwm $hemi.inflated

# mris_curvature -w ?h.white
# mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 $hemi.inflated

mris_sphere -threads 2 $hemi.inflated $hemi.sphere
CURV_ATLAS=$FREESURFER_HOME/average/$hemi.average.curvature.filled.buckner40.tif
# CURV_ATLAS=$FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif
mris_register -curv $hemi.sphere $CURV_ATLAS $hemi.sphere.reg
# done

FSAVG_WHITE=/home/jesperdn/repositories/simnibs-gitlab/simnibs/resources/templates/fsaverage_surf/${hemi}.white.gii
FSAVG_SPHERE=/home/jesperdn/repositories/simnibs-gitlab/simnibs/resources/templates/fsaverage_surf/${hemi}.sphere.gii

if $DO_CAT
then
    echo CAT: $hemi.sphere
    /home/jesperdn/repositories/simnibs-gitlab/simnibs/external/bin/linux/CAT_Surf2Sphere ${hemi}.white ${hemi}.sphere.cat.gii 10

    echo CAT: $hemi.sphere.reg
    /home/jesperdn/repositories/simnibs-gitlab/simnibs/external/bin/linux/CAT_WarpSurf -steps 2 -avg -i ${hemi}.white -is ${hemi}.sphere.cat.gii -t $FSAVG_WHITE -ts $FSAVG_SPHERE -ws ${hemi}.sphere.reg.cat.gii
fi

for hemi in lh rh
do
    if $DO_CAT
    then
        python -c "import cortech; import nibabel as nib; fsa = cortech.Sphere.from_file('/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/surf/$hemi.sphere.reg'); a,b,c=nib.freesurfer.read_annot('/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/label/$hemi.aparc.annot'); sub=cortech.Sphere.from_file('$hemi.sphere.reg'); nv=sub.n_vertices; x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.fs.annot', x,b,c); sub=cortech.Sphere.from_file('epoch-$i.validation.pred.$hemi.registration'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.pred.annot', x,b,c); sub=cortech.Sphere.from_file('/home/jesperdn/repositories/brainsynth/brainsynth/resources/$hemi.sphere.reg'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.map.annot', x[:nv],b,c); sub=cortech.Sphere.from_file('$hemi.sphere.reg.cat.gii'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.cat.annot', x,b,c);"
    else
        python -c "import cortech; import nibabel as nib; fsa = cortech.Sphere.from_file('/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/surf/$hemi.sphere.reg'); a,b,c=nib.freesurfer.read_annot('/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/label/$hemi.aparc.annot'); sub=cortech.Sphere.from_file('$hemi.sphere.reg'); nv=sub.n_vertices; x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.fs.annot', x,b,c); sub=cortech.Sphere.from_file('epoch-$i.validation.pred.$hemi.registration'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.pred.annot', x,b,c); sub=cortech.Sphere.from_file('/home/jesperdn/repositories/brainsynth/brainsynth/resources/$hemi.sphere.reg'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.aparc.map.annot', x[:nv],b,c);"
    fi
done

# hemi=lh

# freeview -f $hemi.white:annot=$hemi.aparc.fs.annot:annot=$hemi.aparc.pred.annot:annot=$hemi.aparc.map.annot

# freeview -f $hemi.white:annot=$hemi.aparc.cat.annot:annot=$hemi.aparc.fs.annot:annot=$hemi.aparc.pred.annot:annot=$hemi.aparc.map.annot