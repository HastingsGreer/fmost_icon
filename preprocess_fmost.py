import os
import argparse
import footsteps
footsteps.initialize(output_root="/playpen-ssd/tgreer/fmost-preprocessed/")

import icon_registration.losses

data_root = "/playpen-raid2/Data/fMost/subject/"
def process(iA, isSeg=False):
    with torch.no_grad():
        iA = iA[None, None, ::2, ::2, ::2]
        iA = iA.cuda().float()
        iA = icon_registration.losses.gaussian_blur(iA, 4, 4)
        iA = torch.nn.functional.interpolate(iA, size=[192, 192, 192], mode="trilinear").cpu()
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
        if "SLA" in name:

            image = torch.tensor(np.asarray(itk.imread(data_root + name[:-1])))

            ds.append(process(image))

            del image

    torch.save(ds, f"{footsteps.output_dir}/fmost_192_{split}.trch")

