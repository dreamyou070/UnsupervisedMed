import os
import argparse
import nibabel as nib
from PIL import Image
def check_img(gz_dir, trg_axis = 100) :
    proxy = nib.load(gz_dir)
    arr = proxy.get_fdata()
    trg = arr[:,:,trg_axis]
    trg_pil = Image.fromarray(trg)
    return trg_pil

def main(args) :

    brain_ex_folder = args.brain_ex_folder
    os.makedirs(brain_ex_folder, exist_ok = True)
    train_folder = os.path.join(brain_ex_folder, 'train')
    os.makedirs(train_folder, exist_ok = True)

    gt_dir = os.path.join(train_folder, 'gt')
    os.makedirs(gt_dir, exist_ok = True)
    rgb_dir = os.path.join(train_folder, 'rgb')
    os.makedirs(rgb_dir, exist_ok=True)
    skull_stripped_dir = os.path.join(train_folder, 'skull_stripped')
    os.makedirs(skull_stripped_dir, exist_ok=True)

    folders = os.listdir(args.base_dir)
    for folder in folders :
        folder_dir = os.path.join(args.base_dir, folder)
        sub_gzs = os.listdir(folder_dir)
        for sub_gz in sub_gzs :
            if 'brainmask' in sub_gz :
                # (3) brain mask dir
                sub_gz_dir = os.path.join(folder_dir, sub_gz)
                for i in range(40, 100) :
                    pil = check_img(sub_gz_dir, trg_axis = i)
                    pil.save(os.path.join(gt_dir, f'{folder}_{i}.png'))
            elif 'brain.nii' in sub_gz :
                # (2) skull stripped
                sub_gz_dir = os.path.join(folder_dir, sub_gz)
                for i in range(40, 100) :
                    pil = check_img(sub_gz_dir, trg_axis = i)
                    pil.save(os.path.join(skull_stripped_dir, f'{folder}_{i}.png'))
            else :
                # (1) full skull
                sub_gz_dir = os.path.join(folder_dir, sub_gz)
                for i in range(40, 100) :
                    pil = check_img(sub_gz_dir, trg_axis = i)
                    pil.save(os.path.join(rgb_dir, f'{folder}_{i}.png'))




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain_ex_folder', type=str,
                        default='/home/dreamyou070/MyData/anomaly_detection/NFBS_Dataset_SY')
    parser.add_argument('--base_dir', type=str,
                        default = '/home/dreamyou070/MyData/anomaly_detection/NFBS_Dataset')
    args=parser.parse_argument()
    main(args)