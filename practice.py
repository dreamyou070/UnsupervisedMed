import os
import nibabel as nib
from PIL import Image

def check_img(gz_dir, trg_axis = 100) :
    proxy = nib.load(gz_dir)
    arr = proxy.get_fdata()
    trg = arr[:,:,trg_axis]
    trg_pil = Image.fromarray(trg)
    return trg_pil

# [1] full skull
full_skull_gz_dir = 'NFBS_Dataset/A00028185/sub-A00028185_ses-NFB3_T1w.nii.gz'
full_skull_pil = check_img(full_skull_gz_dir)
# [2] skull stripped
skull_stripped_gz_dir = 'NFBS_Dataset/A00028185/sub-A00028185_ses-NFB3_T1w_brain.nii.gz'
skull_stripped_pil = check_img(skull_stripped_gz_dir)
# [3] brain mask
brain_mask_gz_dir = 'NFBS_Dataset/A00028185/sub-A00028185_ses-NFB3_T1w_brainmask.nii.gz'
brain_mask_pil = check_img(brain_mask_gz_dir)

full_skull_pil.show()
skull_stripped_pil.show()
brain_mask_pil.show()