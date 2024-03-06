import argparse
import os

def main(args):
    base_folder = 'NFBS_Dataset'
    folders = os.listdir(base_folder)
    for folder in folders :
        if folder == 'A00028185' :
            folder_dir = os.path.join(base_folder, folder)
            trg_folder = os.path.join(folder_dir, 'sub-A00028185_ses-NFB3_T1w.nii')
            trg_file = os.path.join(trg_folder, 'sub-A00028185_ses-NFB3_T1w.nii')
            




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_argument()
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
