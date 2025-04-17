import math

import numpy as np
import torch
from lib.utils.general_utils import matrix_to_quaternion


class BoundingBox:
    def __init__(self, object_type, object_id, size):
        self.object_type = object_type
        self.object_id = object_id

        if isinstance(size, np.ndarray):
            size = torch.from_numpy(size)
        size = size.float().cuda()
        self.size = size  # size_x, size_y, size_z

        self.min_xyz, self.max_xyz = -self.size / 2.0, self.size / 2
        self.frame = {}

    def add_frame_waymo(
        self, frame, metadata, ego2world
    ):  # center_x, center_y, center_z, yaw
        pos = [float(metadata[1]), float(metadata[2]), float(metadata[3])]
        theta = float(metadata[7])

        if isinstance(ego2world, np.ndarray):
            ego2world = torch.from_numpy(ego2world)
        ego2world = ego2world.float().cuda()

        pos = torch.tensor(pos).float().cuda()
        T = ego2world[:3, :3] @ pos + ego2world[:3, 3]

        R = (
            torch.tensor(
                [
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            .float()
            .cuda()
        )
        R = ego2world[:3, :3] @ R

        quaternion: torch.Tensor = matrix_to_quaternion(R)
        quaternion = quaternion / torch.norm(quaternion)
        quaternion = quaternion.unsqueeze(0)

        dT = torch.zeros(3).float().cuda()
        dR = torch.eye(3).float().cuda()
        self.frame[frame] = (T, quaternion, dT, dR)

    def add_frame_kitti(self, frame, transform):
        if isinstance(transform, np.ndarray):
            transform = torch.from_numpy(transform)
        transform = transform.float().cuda()

        pos = transform[:3, 3]
        U, S, V = torch.linalg.svd(transform[:3, :3])

        self.size = torch.max(torch.stack([S, self.size]), dim=0).values
        self.min_xyz, self.max_xyz = -self.size / 2.0, self.size / 2

        quaternion = matrix_to_quaternion(U[:3, :3])
        quaternion = quaternion / torch.norm(quaternion)
        quaternion = quaternion.unsqueeze(0)

        dT = torch.zeros(3).float().cuda()
        dR = torch.eye(3).float().cuda()
        self.frame[frame] = (pos, quaternion, dT, dR)
