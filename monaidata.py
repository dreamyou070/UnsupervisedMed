import os
import tempfile
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import sys
from monai import transforms
from monai.apps import DecathlonDataset, MedNISTDataset

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main() :

    print(f'step 1. setup brats dataset')
    channel = 0  # 0 = Flair
    assert channel in [0, 1, 2, 3], "Choose a valid channel"

    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Lambdad(keys=["image"], func=lambda x: x[channel, :, :, :]),
           #transforms.AddChanneld(keys=["image"]),
            transforms.EnsureTyped(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"], pixdim=(3.0, 3.0, 2.0), mode=("bilinear", "nearest")),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 44)),
            transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 1), random_size=False),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: x.squeeze(-1)),
            transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
            transforms.Lambdad(keys=["slice_label"], func=lambda x: 2.0 if x.sum() > 0 else 1.0),
        ]
    )

    root_dir = './'

    train_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=4,
        download=True,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=train_transforms,
    )
    print(f"Length of training data: {len(train_ds)}")
    print(f'Train image shape {train_ds[0]["image"].shape}')



if __name__ == '__main__' :
    main()