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
from copy import deepcopy
from lib.utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera:
    def __init__(self, timestamp, R, T, w, h, FoVx, FoVy, depth=None, intensity=None,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                ):
        self.timestamp = timestamp
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.depth_map = depth
        self.intensity_map = intensity
        self.image_width = w
        self.image_height = h

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(R, T, trans, scale).T
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def cuda(self):
        cuda_copy = deepcopy(self)
        for k, v in cuda_copy.__dict__.items():
            if isinstance(v, torch.Tensor):
                cuda_copy.__dict__[k] = v.cuda()
        return cuda_copy

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

