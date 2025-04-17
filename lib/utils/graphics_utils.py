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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : torch.Tensor
    color_intensity : torch.Tensor
    normals : torch.Tensor

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt.float()

def getWorld2View2(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = Rt.inverse()
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W.inverse()
    return Rt.float()

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# def world2image(H, W, ir, sensor2world, ego_pose, points : torch.Tensor): # ir is inclination range
#     points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
#     points = points @ sensor2world.T.cuda()
#     points_in_sensor = points[..., :3]

#     azimuth = torch.atan2(points_in_sensor[..., 1], points_in_sensor[..., 0])
#     inclination = torch.asin(points_in_sensor[..., 2] / torch.norm(points_in_sensor, dim=-1))
#     return rays_o, rays_d

def get_rays(K, c2w):
    W, H = int(K[0, 2] * 2), int(K[1, 2] * 2)
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(H - 1, 0, H), indexing='ij')
    i, j = i.t(), j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], torch.ones_like(i)], -1).cuda()
    rays_d = dirs @ c2w.T[:3, :3]
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o.cuda().contiguous(), rays_d.cuda()

def image2point(depthmap, sensor):
    c2w = (sensor.world_view_transform.T).inverse()
    W, H = sensor.image_width, sensor.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ sensor.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def apply_pixel_pose(points, pixel_pose):
    # Extract rotation (roll, pitch, yaw) and translation (x, y, z)
    roll, pitch, yaw = pixel_pose[..., 0], pixel_pose[..., 1], pixel_pose[..., 2]
    translation = pixel_pose[..., 3:]

    rotation_matrix = compute_rotation_matrix(roll, pitch, yaw)

    # Apply the transformation to the points
    transformed_points = torch.einsum('bhwij,bhwj->bhwi', rotation_matrix, points) + translation

    return transformed_points

def compute_rotation_matrix(roll, pitch, yaw):
    # Convert roll, pitch, yaw into rotation matrices
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Compute rotation matrices for roll, pitch, and yaw
    rotation_roll = torch.stack([torch.tensor([1, 0, 0]), torch.stack([0, cos_r, -sin_r], dim=-1), torch.stack([0, sin_r, cos_r], dim=-1)], dim=-1)
    rotation_pitch = torch.stack([torch.stack([cos_p, 0, sin_p], dim=-1), torch.tensor([0, 1, 0]), torch.stack([-sin_p, 0, cos_p], dim=-1)], dim=-1)
    rotation_yaw = torch.stack([torch.stack([cos_y, -sin_y, 0], dim=-1), torch.stack([sin_y, cos_y, 0], dim=-1), torch.tensor([0, 0, 1])], dim=-1)

    # Combine the three rotation matrices
    rotation_matrix = torch.einsum('bij,bjk->bik', rotation_yaw, torch.einsum('bij,bjk->bik', rotation_pitch, rotation_roll))
    return rotation_matrix

def range2point(frame, range_map, sensor):
    '''
    return a tensor of points (H, W, 3) in the world coordinate in HOST
    '''
    ir = sensor.inclination_bounds
    sensor_center = sensor.sensor_center[frame]
    sensor2world = sensor.sensor2world[frame]
    lidar2ego = sensor.sensor2e
    azimuth_correction = torch.atan2(lidar2ego[1, 0], lidar2ego[0, 0])

    # data preprocess
    if not torch.is_tensor(range_map):
        range_map = torch.tensor(range_map, dtype=torch.float32, device='cuda')
    if range_map.dim() != 2:
        if range_map.dim() == 3:
            if range_map.shape[0] == 1:
                range_map = range_map[0]
            elif range_map.shape[2] == 1:
                range_map = range_map[..., 0]
            else:
                raise ValueError('range_map is not (H, W, 1) or (1, H, W)')
        else:
            raise ValueError('range_map shape unindentified')
    if not torch.is_tensor(sensor_center):
        sensor_center = torch.tensor(sensor_center, dtype=torch.float32, device='cuda')
    if not torch.is_tensor(sensor2world):
        sensor2world = torch.tensor(sensor2world, dtype=torch.float32, device='cuda')


    H, W = range_map.shape
    rays_o = sensor_center.cuda()[None, None, ...].expand(H, W, 3)

    y = torch.ones(H, device='cuda', dtype=torch.float32)
    x = (torch.arange(W, 0, -1, device='cuda', dtype=torch.float32) - 0.5) / float(W) #w+1 to avoid the last angle = the first
    grid_y, grid_x = torch.meshgrid(y, x)

    azimuth = grid_x * 2 * torch.pi - torch.pi - azimuth_correction
    if type(ir) != list and type(ir) != tuple:
        ir = [-ir, ir]
    if len(ir) == 2:
        grid_y = grid_y * (torch.arange(H, 0, -1, device='cuda', dtype=torch.float32).unsqueeze(-1) - 0.5) / float(H)
        inclination = grid_y * (ir[1] - ir[0]) + ir[0]
    else:
        inclination = grid_y * torch.tensor((ir), device='cuda', dtype=torch.float32).flip(0).unsqueeze(-1)
    rays_x = torch.cos(inclination) * torch.cos(azimuth)
    rays_y = torch.cos(inclination) * torch.sin(azimuth)
    rays_z = torch.sin(inclination)

    rays_d = torch.stack([rays_x, rays_y, rays_z], dim=-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    points = rays_d * range_map[..., None].cuda()
    points = points @ sensor2world[:3, :3].T.cuda() + sensor2world[:3, 3].cuda()
    return points
