import os
import argparse
import footsteps
footsteps.initialize(output_root="/playpen-ssd/tgreer/fmost-preprocessed/")

import icon_registration.losses
def process(iA, isSeg=False):
    iA = iA[None, None, :, :, :]
    iA = iA.cuda()
    iA = icon_registration.losses.gaussian_blur(iA, 8, 8)
    iA = torch.nn.functional.interpolate(iA, size=[192, 192, 192], mode="trilinear")
    return iA
for split in ["train", "test"]:
    with open(f"splits/{split}.txt") as f:
        image_paths = f.readlines()

    import torch

    import itk
    import tqdm
    import numpy as np
    import glob

    ds = []

    for name in tqdm.tqdm(list(iter(image_paths))[:]):

        image = torch.tensor(np.asarray(itk.imread(name)))

        ds.append(process(image))

    torch.save(ds, f"{footsteps.output_dir}/fmost_192_{split}.trch")

