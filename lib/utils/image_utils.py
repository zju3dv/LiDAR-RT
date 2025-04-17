#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import cv2
import matplotlib


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def color_mapping(tensor, colormap_, reversed=False):

    # To numpy
    tensor_type = ""
    if isinstance(tensor, torch.Tensor):
        tensor_type = "torch_cuda" if tensor.is_cuda else "torch_cpu"
        tensor = tensor.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        tensor_type = "numpy"
    # Normalize
    tensor = tensor.astype(np.float32)
    org_shape = tensor.shape
    tensor = tensor.reshape(-1)
    if tensor.min() != 0 or tensor.max() != 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    if reversed:
        tensor = 1 - tensor
    # mapping

    if isinstance(colormap_, int):
        color = cv2.cvtColor(cv2.applyColorMap(np.uint8(tensor * 255), colormap_), cv2.COLOR_BGR2RGB)
        color =  np.array(color[:, 0, :]).astype(np.float32)  / 255.
    elif isinstance(colormap_, matplotlib.colors.Colormap):
        color = colormap_(tensor)
        color = color[:, 0:3] * color[:, 3:4] + (1 - color[:, 3:4])

    new_shape = list(org_shape)
    new_shape.append(3)
    color = color.reshape(new_shape)
    # Turn back
    if tensor_type == "torch_cuda":
        color = torch.from_numpy(color).cuda()
    elif tensor_type == "torch_cpu":
        color = torch.from_numpy(color)

    return color
