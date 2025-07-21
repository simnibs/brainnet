# Check TopoFit prediction
# ------------------------
MODEL=synth-random
CHECKPOINT=00780
DATASET=ADNI-GO2
SUBID=sub-241346

TRUE_DIR=$CORTECH/nobackup/training_data/full
PRED_DIR=$SCRATCH/results/TopoFit/$MODEL/evaluation/validation/checkpoint-$CHECKPOINT

freeview \
    $TRUE_DIR/$DATASET/$SUBID/T1w.nii \
    $TRUE_DIR/$DATASET/$SUBID/FLAIR.nii \
    --edgecolor red -f \
    $TRUE_DIR/$DATASET/$SUBID/*h.white \
    $TRUE_DIR/$DATASET/$SUBID/*h.pial \
    --edgecolor yellow -f \
    $PRED_DIR/$DATASET/$SUBID/*h.white \
    $PRED_DIR/$DATASET/$SUBID/*h.pial



# Check T1w - FLAIR coregistration

SUBID=sub-432781

freeview \
    $CORTECH/nobackup/training_data/full/ADNI-GO2/$SUBID/T1w.nii \
    $CORTECH/nobackup/training_data/full/ADNI-GO2/$SUBID/FLAIR.nii \
    -f $CORTECH/nobackup/training_data/full/ADNI-GO2/$SUBID/*h.white \
    $CORTECH/nobackup/training_data/full/ADNI-GO2/$SUBID/*h.pial \
    *h.white *h.pial


# Check subject2subject registration
model="synth-random"
checkpoint="00780"
ROOT_PRED=$SCRATCH/results/TopoFit/$model/evaluation/validation/checkpoint-$checkpoint
ROOT_TRUE=$CORTECH/nobackup/training_data/full

SUB=sub-236974
freeview \
    $ROOT_TRUE/ADNI-GO2/$SUB/T1w.nii $ROOT_TRUE/ADNI-GO2/$SUB/FLAIR.nii \
    -f $ROOT_PRED/ADNI-GO2/$SUB/PRED/lh.white \
    $ROOT_PRED/ADNI-GO2/$SUB/PRED/lh.white.resampled_on_true


gzip -dk *.nii.gz
mv *.nii ../../ADNI-GO2/

for f in *.nii
    d=${f%.nii}
    mkdir -p $d
    mv $f $d/FLAIR.nii
done

for f in *.nii; do d=${f%.nii} && mkdir -p $d && mv $f $d/T1w.nii; done


freeview \
    $CORTECH/nobackup/training_data/full/ADNI-GO2/$SUB/T1w.nii $CORTECH/nobackup/training_data/full/ADNI-GO2/$SUB/FLAIR.nii \
    -f $SCRATCH/results/TopoFit/synth-random/evaluation/validation/checkpoint-00800/ADNI-GO2/$SUB/*h.white $SCRATCH/results/TopoFit/synth-random/evaluation/validation/checkpoint-00800/ADNI-GO2/$SUB/*h.pial $SCRATCH/results/TopoFit/synth-random-clinical/evaluation/validation/checkpoint-00800/ADNI-GO2/$SUB/*h.white $SCRATCH/results/TopoFit/synth-random-clinical/evaluation/validation/checkpoint-00800/ADNI-GO2/$SUB/*h.pial $SCRATCH/results/TopoFit/synth-random-clinical-axial/evaluation/validation/checkpoint-00800/ADNI-GO2/$SUB/*h.white $SCRATCH/results/TopoFit/synth-random-clinical-axial/evaluation/validation/checkpoint-00800/ADNI-GO2/$SUB/*h.pial $CORTECH/nobackup/training_data/full/ADNI-GO2/$SUB/*h.white $CORTECH/nobackup/training_data/full/ADNI-GO2/$SUB/*h.pial



# RSYNC

# sync subset of files

SOURCE_DIR=
DEST_DIR=
# exclude everything; then include specific items
rsync -auvh --prune-empty-dirs \
    # include subdirectories
    --include="*/" \
    --include="*h.white" \
    --exclude="*" \
    $SOURCE_DIR $DEST_DIR -n

