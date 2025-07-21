
for hemi in lh rh
do
    echo $hemi
    echo "-------"
    if $DO_CAT
    then
        python -c "import nibabel as nib; import numpy as np; a,ctab,names = nib.freesurfer.read_annot('$hemi.aparc.fs.annot');b = nib.freesurfer.read_annot('$hemi.aparc.pred.annot')[0];c = nib.freesurfer.read_annot('$hemi.aparc.map.annot')[0];d = nib.freesurfer.read_annot('$hemi.aparc.cat.annot')[0];print('DICE SCORES\n');print('overall\n');print('PRED', (a==b).sum() / len(a));print('MAP ', (a==c).sum() / len(a));print('CAT ', (a==d).sum() / len(a)); res = [(i,names[j].decode(),2 * ((a==i) & (b==i)).sum() / (np.sum(a==i) + np.sum(b==i)), 2 * ((a==i) & (c==i)).sum() / (np.sum(a==i) + np.sum(c==i)), 2 * ((a==i) & (d==i)).sum() / (np.sum(a==i) + np.sum(d==i)), np.sum(a==i)/len(a.ravel())*100.0) for j,i in enumerate(np.unique(a))]; print('                                       PRED    MAP   CAT    % area'); [print(f'{r[0]:02d}    {r[1]:30s}   {r[2]:4.2f} {\" <\" if r[2] < r[3] else \"> \":s} {r[3]:4.2f}  {r[4]:4.2f} {r[5]:5.1f}') for r in res]"
    else
        python -c "import nibabel as nib; import numpy as np; a,ctab,names = nib.freesurfer.read_annot('$hemi.aparc.fs.annot');b = nib.freesurfer.read_annot('$hemi.aparc.pred.annot')[0];c = nib.freesurfer.read_annot('$hemi.aparc.map.annot')[0];print('DICE SCORES\n');print('overall\n');print('PRED', (a==b).sum() / len(a));print('MAP ', (a==c).sum() / len(a));res = [(i,names[j].decode(),2 * ((a==i) & (b==i)).sum() / (np.sum(a==i) + np.sum(b==i)), 2 * ((a==i) & (c==i)).sum() / (np.sum(a==i) + np.sum(c==i)), np.sum(a==i)/len(a.ravel())*100.0) for j,i in enumerate(np.unique(a))]; print('                                       PRED    MAP   % area'); [print(f'{r[0]:02d}    {r[1]:30s}   {r[2]:4.2f} {\" <\" if r[2] < r[3] else \"> \":s} {r[3]:4.2f}  {r[4]:4.2f}') for r in res]"
    fi
done


hemi=lh

freeview -f $hemi.white:annot=$hemi.aparc.cat.annot:annot=$hemi.aparc.fs.annot:annot=$hemi.aparc.pred.annot:annot=$hemi.aparc.map.annot