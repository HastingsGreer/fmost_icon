#!/usr/bin/env python

import argparse

import cvpr_network

import icon_registration as icon
import icon_registration.itk_wrapper
import itk
import numpy as np
import torch
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", required=True)
parser.add_argument("--fixed_image", required=True)
parser.add_argument("--moving_image", required=True)
parser.add_argument("--transform_out")
parser.add_argument("--displacement_image_out")
parser.add_argument("--warped_moving_out")
parser.add_argument("--finetune", action="store_true")

args = parser.parse_args()

net = cvpr_network.make_network([1, 1, 192, 192, 192])

if args.weights_path:
    weights = torch.load(args.weights_path)
    net.regis_net.load_state_dict(weights)

fixed_image = itk.imread(args.fixed_image)
print(type(fixed_image))
moving_image = itk.imread(args.moving_image)

def brain_network_preprocess(image: "itk.Image") -> "itk.Image":
    if type(image) == itk.Image[itk.UC, 3] :
        cast_filter = itk.CastImageFilter[itk.Image[itk.UC, 3], itk.Image[itk.F, 3]].New()
        cast_filter.SetInput(image)
        cast_filter.Update()
        image = cast_filter.GetOutput()
    _, max_ = itk.image_intensity_min_max(image)
    image = itk.shift_scale_image_filter(image, shift=0., scale = 1. / max_)
    return image

fixed_image = brain_network_preprocess(fixed_image)
moving_image = brain_network_preprocess(moving_image)

# We want images normalized to 0.0, 1.0
for image in fixed_image, moving_image:
    assert np.abs(np.max(np.array(image)) - 1.0) < 0.2
    assert np.abs(np.min(np.array(image))) < 0.2

phi, _ = icon.itk_wrapper.register_pair(
    net, moving_image, fixed_image, finetune_steps=90 if args.finetune else None
)

if args.transform_out:
    itk.transformwrite([phi], args.transform_out)

if args.displacement_image_out:
    filter = itk.TransformToDisplacementFieldFilter[
        itk.itkImagePython.itkImageVF33, itk.D
    ].New()
    decorator = itk.DataObjectDecorator[itk.Transform[itk.D, 3, 3]].New()
    decorator.Set(phi)
    filter.SetInput(decorator)
    filter.SetReferenceImage(fixed_image)
    filter.SetUseReferenceImage(True)
    filter.Update()
    itk.imwrite(filter.GetOutput(), args.displacement_image_out)

if args.warped_moving_out:
    warped = itk.resample_image_filter(
        moving_image,
        use_reference_image=True,
        reference_image=fixed_image,
        transform=phi,
        interpolator=itk.LinearInterpolateImageFunction.New(moving_image),
    )
    itk.imwrite(warped, args.warped_moving_out)
