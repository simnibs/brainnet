#!/bin/bash

#SBATCH --job-name=sub2sub         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=4G                   # Job memory request
#SBATCH --array=1-489          # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

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

model="synth-random-reg"
checkpoint="best"
ROOT_PRED=$SCRATCH/results/TopoFit/$model/evaluation/validation/checkpoint-$checkpoint
ROOT_TRUE=$CORTECH/nobackup/training_data/full

SUB_DIR=$(find $ROOT_PRED -maxdepth 2 -type d -name "*sub-*" | sed -n "${SLURM_ARRAY_TASK_ID}p")
subid=$(basename $SUB_DIR)
dataset=$(basename $(dirname $SUB_DIR))

cd $SUB_DIR

echo Running FreeSurfer commands in $PWD

mkdir -p ./sphere_true
mkdir -p ./sphere_pred

# Create spherical regs using FS
for hemi in lh rh
do
    CURV_ATLAS=$FREESURFER_HOME/average/$hemi.average.curvature.filled.buckner40.tif

    # sphere_true
    # =====================
    cd sphere_true
    cp $ROOT_TRUE/$dataset/$subid/$hemi.white .
    cp $ROOT_TRUE/$dataset/$subid/$hemi.pial .

    # steps taken from recon-all dev table
    mris_smooth -n 3 -nw $hemi.white $hemi.smoothwm
    mris_inflate $hemi.smoothwm $hemi.inflated
    mris_curvature -w $hemi.white
    mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 $hemi.inflated
    mris_sphere $hemi.inflated $hemi.sphere

    cd ..

    # sphere_pred
    # =====================
    cd sphere_pred

    cp ../$hemi.white .
    cp ../$hemi.pial .

    # steps taken from recon-all dev table
    mris_smooth -n 3 -nw $hemi.white $hemi.smoothwm
    mris_inflate $hemi.smoothwm $hemi.inflated
    mris_curvature -w $hemi.white
    mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 $hemi.inflated
    mris_sphere $hemi.inflated $hemi.sphere

    # register to FSAVERAGE
    mris_register -curv $hemi.sphere $CURV_ATLAS $hemi.sphere.reg

    cd ..

    # register sphere_true to sphere_pred
    # =====================
    mris_register -1 -curv ./sphere_true/$hemi.sphere ./sphere_pred/$hemi.sphere ./sphere_true/$hemi.sphere.reg2pred
done

# Resample the true surface to the predicted surface
for hemi in lh rh
do
    TARGET_REG=./sphere_pred/$hemi.sphere
    SOURCE_REG=./sphere_true/$hemi.sphere.reg2pred
    for surf in white pial
    do
        SOURCE_SURF=./sphere_true/$hemi.$surf
        OUT=./sphere_pred/$hemi.$surf.resampled_on_true
        mris_resample --atlas_reg $TARGET_REG \
            --subject_reg $SOURCE_REG --subject_surf $SOURCE_SURF \
            --out $OUT
    done

    # resample sphere_pred on sphere_true
    #python -c "import cortech; import nibabel as nib; pred = cortech.SphericalRegistration.from_freesurfer('./sphere_pred/$hemi.sphere'); true = cortech.SphericalRegistration.from_freesurfer('./sphere_true/$hemi.sphere.reg2pred'); true_surf = cortech.Surface.from_freesurfer('./sphere_true/$hemi.white'); pred.faces = pred.faces.astype(int); true.faces = true.faces.astype(int); true.project(pred); v = true.resample(true_surf.vertices); nib.freesurfer.write_geometry('./sphere_pred/$hemi.white.resampled_on_true', v, pred.faces, volume_info=true_surf.sphere_pred.geometry.as_freesurfer_dict()); true_surf = cortech.Surface.from_freesurfer('./sphere_true/$hemi.pial'); true.faces = true.faces.astype(int); v = true.resample(true_surf.vertices); nib.freesurfer.write_geometry('./sphere_pred/$hemi.pial.resampled_on_true', v, pred.faces, volume_info=true_surf.sphere_pred.geometry.as_freesurfer_dict());"

    # resample APARC on sphere_pred
    #python -c "import nibabel as nib; import cortech; label, ctab, names = nib.freesurfer.read_annot('$FS_APARC'); fsavg = cortech.SphericalRegistration.from_freesurfer('$FS_SURF'); subject = cortech.SphericalRegistration.from_freesurfer('./sphere_pred/$hemi.sphere.reg'); fsavg.project(subject, method='nearest'); subject_label = fsavg.resample(label); nib.freesurfer.write_annot('./sphere_pred/${hemi}.aparc.annot', subject_label, ctab, names);"

    # Compute vertex to vertex distances

    # white
    python -c "import numpy as np; import nibabel as nib; import cortech; p = cortech.Surface.from_freesurfer('./$hemi.white'); p2t = cortech.Surface.from_freesurfer('./sphere_pred/$hemi.white.resampled_on_true'); p2t.to_scanner_ras(); dist = np.linalg.norm(p2t.vertices - p.vertices, axis=1); nib.freesurfer.write_morph_data('./sphere_pred/${hemi}.white.distance', dist);
    "
    # gray
    python -c "import numpy as np; import nibabel as nib; import cortech; p = cortech.Surface.from_freesurfer('./$hemi.pial'); p2t = cortech.Surface.from_freesurfer('./sphere_pred/$hemi.pial.resampled_on_true'); p2t.to_scanner_ras(); dist = np.linalg.norm(p2t.vertices - p.vertices, axis=1); nib.freesurfer.write_morph_data('./sphere_pred/${hemi}.pial.distance', dist);
    "
done

echo Running CAT
cd ./sphere_pred
for hemi in lh rh
do
    FSAVG_WHITE=/home/jesperdn/repositories/simnibs-gitlab/simnibs/resources/templates/fsaverage_surf/${hemi}.white.gii
    FSAVG_SPHERE=/home/jesperdn/repositories/simnibs-gitlab/simnibs/resources/templates/fsaverage_surf/${hemi}.sphere.gii

    echo CAT: $hemi.sphere
    /home/jesperdn/repositories/simnibs-gitlab/simnibs/external/bin/linux/CAT_Surf2Sphere ${hemi}.white ${hemi}.cat.sphere.gii 10

    echo CAT: $hemi.sphere.reg
    /home/jesperdn/repositories/simnibs-gitlab/simnibs/external/bin/linux/CAT_WarpSurf -steps 2 -avg -i ${hemi}.white -is ${hemi}.cat.sphere.gii -t $FSAVG_WHITE -ts $FSAVG_SPHERE -ws ${hemi}.cat.sphere.reg.gii
done
cd ..

echo Running JOSA
cd ./sphere_pred
for hemi in lh rh
do
    josareg --surfdir . --hemi $hemi --o . --no-link
done
cd ..

# map APARC using different methods
cd ./sphere_pred
for hemi in lh rh
do
    # resample APARC from fsaverage to subject using the different methods
    FS_APARC=$FREESURFER_HOME/subjects/fsaverage/label/$hemi.aparc.annot
    FS_REG=$FREESURFER_HOME/subjects/fsaverage/surf/$hemi.sphere.reg
    FS_SURF=$FREESURFER_HOME/subjects/fsaverage/surf/$hemi.white
    TMPFILE=./tmpfile

    TARGET_REGS=(
        "/home/jesperdn/repositories/brainsynth/brainsynth/resources/$hemi.sphere.reg"
        "../$hemi.registration"
        "./$hemi.sphere.reg"
        # "./$hemi.cat.sphere.reg.gii"
        "./$hemi.josa.sphere.reg"
    )
    SUB_ANNOTS=(
        "./${hemi}.map.aparc.annot"
        "./${hemi}.pred.aparc.annot"
        "./${hemi}.aparc.annot"
        # "./${hemi}.cat.aparc.annot"
        "./${hemi}.josa.aparc.annot"
    )

    for i in "${!TARGET_REGS[@]}"
    do
        TARGET_REG=${TARGET_REGS[i]}
        SUB_ANNOT=${SUB_ANNOTS[i]}
        echo Target reg: $TARGET_REG
        echo output    : $SUB_ANNOT
        mris_resample --atlas_reg $TARGET_REG \
            --subject_reg $FS_REG --subject_surf $FS_SURF \
            --out $TMPFILE --annot_in $FS_APARC --annot_out $SUB_ANNOT
        rm $TMPFILE
    done

    echo CAT...
    python -c "import cortech; import nibabel as nib; fsa = cortech.Sphere.from_file('$FREESURFER_HOME/subjects/fsaverage/surf/$hemi.sphere.reg'); a,b,c=nib.freesurfer.read_annot('$FREESURFER_HOME/subjects/fsaverage/label/$hemi.aparc.annot'); sub=cortech.Sphere.from_file('$hemi.cat.sphere.reg.gii'); x = fsa.project_and_resample(sub, a, method='nearest'); nib.freesurfer.write_annot('$hemi.cat.aparc.annot', x,b,c);"
done
cd ..
