#!/bin/bash

#SBATCH --job-name=sphericalreg         # Job name
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

echo $sub

cd $sub

for hemi in lh rh
do
    echo $hemi

    torchsurf=$sub/${hemi}.white.6.prediction.pt
    insurf=$sub/freesurfer.${hemi}.white.6.prediction
    sphere=$sub/${hemi}.sphere
    fssphere=$sub/freesurfer.${hemi}.sphere
    spherereg=$sub/${hemi}.sphere.reg

    if [[ ! -f ${spherereg}.pt ]]; then

        echo "$hemi: spherical registration of $sub"

        python /mrhome/jesperdn/repositories/brainsynth/tools/torch_vertices_to_freesurfer_surface.py ${torchsurf} $hemi
        python /mrhome/jesperdn/repositories/brainsynth/tools/torch_vertices_to_freesurfer_surface.py ${sphere}.pt $hemi

        ln -s $insurf smoothwm
        ln -s $fssphere sphere

        mris_register -curv sphere $FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif $spherereg

        python -c "import nibabel as nib; import torch; v, _ = nib.freesurfer.read_geometry('$spherereg'); torch.save(torch.tensor(v, dtype=torch.float), '${spherereg}.pt')"

        rm smoothwm sphere $spherereg $fssphere $insurf
    fi
done


# hemi=lh

# torchsurf=$sub/${hemi}.white.6.prediction.pt
# insurf=$sub/freesurfer.${hemi}.white.6.prediction
# sphere=$sub/${hemi}.sphere
# fssphere=$sub/freesurfer.${hemi}.sphere
# spherereg=$sub/${hemi}.sphere.reg

# if [[ ! -f ${spherereg}.pt ]]; then

#     echo "$hemi: spherical registration of $sub"

#     python /mrhome/jesperdn/repositories/brainsynth/tools/torch_vertices_to_freesurfer_surface.py ${torchsurf} $hemi
#     python /mrhome/jesperdn/repositories/brainsynth/tools/torch_vertices_to_freesurfer_surface.py ${sphere}.pt $hemi

#     ln -s $insurf smoothwm
#     ln -s $fssphere sphere

#     mris_register -curv sphere $FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif $spherereg

#     python -c "import nibabel as nib; import torch; v, _ = nib.freesurfer.read_geometry('$spherereg'); torch.save(torch.tensor(v, dtype=torch.float), '${spherereg}.pt')"

#     rm smoothwm sphere $spherereg $fssphere $insurf
# fi

# # Right

# hemi=rh

# torchsurf=$sub/${hemi}.white.6.prediction.pt
# insurf=$sub/freesurfer.${hemi}.white.6.prediction
# sphere=$sub/${hemi}.sphere
# fssphere=$sub/freesurfer.${hemi}.sphere
# spherereg=$sub/${hemi}.sphere.reg

# if [[ ! -f ${spherereg}.pt ]]; then

#     echo "$hemi: spherical registration of $sub"

#     python /mrhome/jesperdn/repositories/brainsynth/tools/torch_vertices_to_freesurfer_surface.py ${torchsurf} $hemi
#     python /mrhome/jesperdn/repositories/brainsynth/tools/torch_vertices_to_freesurfer_surface.py ${sphere}.pt $hemi

#     ln -s $insurf smoothwm
#     ln -s $fssphere sphere

#     mris_register -curv sphere $FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif $spherereg

#     python -c "import nibabel as nib; import torch; v, _ = nib.freesurfer.read_geometry('$spherereg'); torch.save(torch.tensor(v, dtype=torch.float), '${spherereg}.pt')"

#     rm smoothwm sphere $spherereg $fssphere $insurf

# fi