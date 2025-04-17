import math
from copy import deepcopy

import numpy as np
import torch
from lib.scene.cameras import Camera
from lib.utils.graphics_utils import (
    apply_pixel_pose,
    getProjectionMatrix,
    getWorld2View2,
)
from lib.utils.other_utils import depth2normal
from torch import nn

ego2cam = torch.tensor(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=torch.float32
)


class LiDARSensor:
    def __init__(self, sensor2ego, name, inclination_bounds, data_type):
        if isinstance(sensor2ego, np.ndarray):
            sensor2ego = torch.from_numpy(sensor2ego)
        sensor2ego = sensor2ego.float().cpu()

        self.sensor2ego = sensor2ego  # e2w
        self.sensor_center = {}  # key: frame, value: tensor(3)
        self.sensor2world = {}  # key: frame, value: tensor(4, 4)
        self.ego2world = {}
        self.range_image_return1 = {}  # key: frame, value: tensor(H, W, 4)
        self.range_image_return2 = {}  # key: frame, value: tensor(H, W, 4)
        self.pixel_pose = {}
        self.H, self.W = 0, 0
        self.num_frames = 0
        self.train_frames = []
        self.eval_frames = []

        self.name = name
        self.inclination_bounds = inclination_bounds
        self.data_type = data_type

        if self.data_type == "Waymo":
            self.pixel_offset = 0.5
            self.angle_offset = torch.atan2(
                self.sensor2ego[1, 0], self.sensor2ego[0, 0]
            )
        elif self.data_type == "KITTI":
            self.pixel_offset = 0.0
            self.angle_offset = 0.0
        else:
            raise ValueError("Could not recongnize the data type")

    def get_mask(self, frame, return_num=1):
        if return_num == 1:
            return self.range_image_return1[frame][..., 0] != 0
        else:
            return self.range_image_return2[frame][..., 0] != 0

    def get_depth(self, frame, return_num=1):
        if return_num == 1:
            return self.range_image_return1[frame][..., 0]
        else:
            return self.range_image_return2[frame][..., 0]

    def get_intensity(self, frame, return_num=1):
        if return_num == 1:
            return self.range_image_return1[frame][..., 1]
        else:
            return self.range_image_return2[frame][..., 1]

    def get_normal(self, frame, return_num=1):
        depth = self.get_depth(frame)
        rayhit_mask = self.get_mask(frame)
        normal = depth2normal(depth, frame, self)
        valid_mask = (rayhit_mask[2:, 1:-1] & rayhit_mask[:-2, 1:-1]) & (
            rayhit_mask[1:-1, 2:] & rayhit_mask[1:-1, :-2]
        )
        valid_mask = torch.nn.functional.pad(
            valid_mask, (1, 1, 1, 1), mode="constant", value=0
        )
        return normal, valid_mask

    def set_frames(self, train_frames, eval_frames):
        self.train_frames = train_frames
        self.eval_frames = eval_frames
        print("train:", train_frames)
        print("eval:", eval_frames)
        assert (
            len(self.train_frames) + len(self.eval_frames) <= self.num_frames
        ), "Found illegal frame ranges!"

    def add_frame(self, frame, ego2world, r1, r2, pixel_pose=None):
        if isinstance(ego2world, np.ndarray):
            ego2world = torch.from_numpy(ego2world)
        if isinstance(r1, np.ndarray):
            r1 = torch.from_numpy(r1)
        if isinstance(r2, np.ndarray):
            r2 = torch.from_numpy(r2)

        sensor2world = ego2world.float().cpu() @ self.sensor2ego
        sensor_center = sensor2world[:3, 3]
        r1 = r1.float().cpu()
        r2 = r2.float().cpu()

        self.sensor_center[frame] = sensor_center
        self.sensor2world[frame] = sensor2world
        self.ego2world[frame] = ego2world
        self.range_image_return1[frame] = r1
        self.range_image_return2[frame] = r2
        self.num_frames += 1

        if pixel_pose is not None:
            if isinstance(pixel_pose, np.ndarray):
                pixel_pose = torch.from_numpy(pixel_pose)
            self.pixel_pose[frame] = pixel_pose.float().cpu()

        if self.H == 0 and self.W == 0:
            self.H, self.W, _ = self.range_image_return1[frame].shape
        elif (
            self.H != self.range_image_return1[frame].shape[0]
            or self.W != self.range_image_return1[frame].shape[1]
        ):
            raise ValueError("range image size mismatch")

    def inverse_projection(self, frame, pixel_pose=False):
        """
        Range image to world coordinates.

        Args:
            frame: Input frame.
            pixel_pose: Tensor of shape [H, W, 6] representing [roll, pitch, yaw, x, y, z]
                        for each pixel in the range image, which defines the transform
                        from the vehicle frame to the global frame.

        Returns:
            pts: Tensor of shape (N, 3) in CPU containing the world coordinates of the points.
            lidar_intensity: Tensor of shape (N) in CPU containing the intensity values.
        """
        sensor2world = self.sensor2world[frame]
        sensor_center = self.sensor_center[frame]

        lidar_pts_r1 = self.range_image_return1[frame][..., 0]
        lidar_intensity_r1 = self.range_image_return1[frame][..., 1]
        pts_r1 = self.range2point(frame, lidar_pts_r1)

        if pixel_pose:
            pixel_pose_r1 = self.pixel_pose[frame]
            pts_r1 = apply_pixel_pose(pts_r1, pixel_pose_r1)

        mask = lidar_intensity_r1 != -1
        pts_r1, lidar_intensity_r1 = pts_r1[mask], lidar_intensity_r1[mask]

        lidar_pts_r2 = self.range_image_return2[frame][..., 0]
        lidar_intensity_r2 = self.range_image_return2[frame][..., 1]
        pts_r2 = self.range2point(frame, lidar_pts_r2)

        if pixel_pose:
            pixel_pose_r2 = self.pixel_pose[frame]
            pts_r2 = apply_pixel_pose(pts_r2, pixel_pose_r2)

        mask = lidar_intensity_r2 != -1
        pts_r2, lidar_intensity_r2 = pts_r2[mask], lidar_intensity_r2[mask]

        pts = torch.cat([pts_r1, pts_r2], dim=0).cpu()
        lidar_intensity = torch.cat(
            [lidar_intensity_r1, lidar_intensity_r2], dim=0
        ).cpu()
        return pts.view(-1, 3), lidar_intensity.view(-1)

    def inverse_projection_with_range(self, frame, range_map, mask):
        """
        Range image to world coordinates.

        Args:
            @frame: Input frame.
            @range_map: (H, W, 1)
            @mask: (H, W) or (H, W, 1)
        Returns:
            pts: (N, 3)
        """

        pts = self.range2point(frame, range_map)
        pts = torch.tensor(pts, device="cuda")
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().to(pts.device)
        if len(mask.shape) == 2:
            pts = pts[mask]
        elif len(mask.shape) == 3:
            pts = pts * mask

        return pts.view(-1, 3)

    def fuse_range_image(self, pixel_pose=False):
        P, T = [], []
        for frame in sorted(self.sensor_center.keys()):
            pts, lidar_intensity = self.inverse_projection(frame, pixel_pose)
            P.append(pts)
            T.append(lidar_intensity)

        pts = torch.cat(P, dim=0)
        lidar_intensity = torch.cat(T, dim=0)
        return pts.view(-1, 3), lidar_intensity.view(-1)

    def gen_cam(self, frame, batch_size=8, height=800, width=1200, focal=800):
        radians = torch.rand(batch_size) * 2 * math.pi
        sin = torch.sin(radians)
        cos = torch.cos(radians)
        R = torch.zeros(batch_size, 3, 3)
        R[:, 0, 0], R[:, 0, 2] = cos, -sin
        R[:, 1, 1] = torch.ones_like(R[:, 1, 1])
        R[:, 2, 0], R[:, 2, 2] = sin, cos
        R = (
            R
            @ ego2cam.unsqueeze(0)
            @ self.sensor2world[frame][:3, :3].inverse().unsqueeze(0)
        )  # w2c
        T = -R @ self.sensor_center[frame]
        train_cameras = []
        for i in range(batch_size):
            camera = Camera(
                timestamp=frame,
                R=R[i].transpose(0, 1),
                T=T[i],
                w=width,
                h=height,
                FoVx=2 * math.atan(width / (2 * focal)),
                FoVy=2 * math.atan(height / (2 * focal)),
            )
            points, intensity = self.inverse_projection(frame)
            points = (
                torch.cat([points, torch.ones((points.shape[0], 1))], dim=1)
                .float()
                .cuda()
            )
            points_camera = points @ camera.world_view_transform.cuda()
            points_proj = points_camera @ camera.projection_matrix.cuda()
            points_proj = points_proj[:, :3] / points_proj[:, 3, None]

            uvz = torch.zeros_like(points).cuda().float()
            uvz[:, 0] = ((points_proj[:, 0] + 1.0) * camera.image_width - 1) * 0.5
            uvz[:, 1] = ((points_proj[:, 1] + 1.0) * camera.image_height - 1) * 0.5
            uvz[:, 2] = points_camera[:, 2]
            mask = (
                (uvz[:, 2] > 0)
                & (uvz[:, 1] >= 0)
                & (uvz[:, 1] < camera.image_height)
                & (uvz[:, 0] >= 0)
                & (uvz[:, 0] < camera.image_width)
            )
            uvz[:, 2] = torch.norm(points_camera[:, :3], dim=1)
            uvz = uvz[mask]
            uv = uvz[:, :2].to(torch.int32)

            depth_map = torch.zeros(
                [1, camera.image_height, camera.image_width], device="cuda"
            )
            depth_map[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]

            camera.depth_map = depth_map.float().cpu()

            intensity_map = torch.zeros(
                [1, camera.image_height, camera.image_width], device="cuda"
            )
            intensity_map[0, uv[:, 1], uv[:, 0]] = intensity.cuda()[mask]

            camera.intensity_map = intensity_map.float().cpu()

            train_cameras.append(camera)
        return train_cameras

    def gen_norot_cam(self, frame, height=800, width=1200, focal=800):
        radians = 0
        sin = math.sin(radians)
        cos = math.cos(radians)
        R = torch.zeros(3, 3)
        R[0, 0], R[0, 2] = cos, -sin
        R[1, 1] = 1
        R[2, 0], R[2, 2] = sin, cos
        R = R @ ego2cam @ self.sensor2world[frame][:3, :3].inverse()  # w2c
        T = -R @ self.sensor_center[frame]
        camera = Camera(
            timestamp=frame,
            R=R.transpose(0, 1),
            T=T,
            w=width,
            h=height,
            FoVx=2 * math.atan(width / (2 * focal)),
            FoVy=2 * math.atan(height / (2 * focal)),
        )
        points, intensity = self.inverse_projection(frame)
        points = (
            torch.cat([points, torch.ones((points.shape[0], 1))], dim=1).float().cuda()
        )
        points_camera = points @ camera.world_view_transform.cuda()
        points_proj = points_camera @ camera.projection_matrix.cuda()
        points_proj = points_proj[:, :3] / points_proj[:, 3, None]

        uvz = torch.zeros_like(points).cuda().float()
        uvz[:, 0] = ((points_proj[:, 0] + 1.0) * camera.image_width - 1) * 0.5
        uvz[:, 1] = ((points_proj[:, 1] + 1.0) * camera.image_height - 1) * 0.5
        uvz[:, 2] = points_camera[:, 2]
        mask = (
            (uvz[:, 2] > 0)
            & (uvz[:, 1] >= 0)
            & (uvz[:, 1] < camera.image_height)
            & (uvz[:, 0] >= 0)
            & (uvz[:, 0] < camera.image_width)
        )
        uvz = uvz[mask]
        uv = uvz[:, :2].to(torch.int32)

        depth_map = torch.zeros(
            [1, camera.image_height, camera.image_width], device="cuda"
        )
        depth_map[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
        camera.depth_map = depth_map.detach().float().cpu()
        intensity_map = torch.zeros(
            [1, camera.image_height, camera.image_width], device="cuda"
        )
        intensity_map[0, uv[:, 1], uv[:, 0]] = intensity.cuda()[mask]
        camera.intensity_map = intensity_map.detach().float().cpu()

        return camera

    def range2point(self, frame, range_map):
        """
        return a tensor of points (H, W, 3) in the world coordinate in HOST
        """
        ir = self.inclination_bounds
        sensor_center = self.sensor_center[frame]
        sensor2world = self.sensor2world[frame]

        # data preprocess
        if not torch.is_tensor(range_map):
            range_map = torch.tensor(range_map, dtype=torch.float32, device="cuda")
        if range_map.dim() != 2:
            if range_map.dim() == 3:
                if range_map.shape[0] == 1:
                    range_map = range_map[0]
                elif range_map.shape[2] == 1:
                    range_map = range_map[..., 0]
                else:
                    raise ValueError("range_map is not (H, W, 1) or (1, H, W)")
            else:
                raise ValueError("range_map shape unindentified")
        if not torch.is_tensor(sensor_center):
            sensor_center = torch.tensor(
                sensor_center, dtype=torch.float32, device="cuda"
            )
        if not torch.is_tensor(sensor2world):
            sensor2world = torch.tensor(
                sensor2world, dtype=torch.float32, device="cuda"
            )

        H, W = range_map.shape
        rays_o = sensor_center.cuda()[None, None, ...].expand(H, W, 3)

        y = torch.ones(H, device="cuda", dtype=torch.float32)
        x = (
            torch.arange(W, 0, -1, device="cuda", dtype=torch.float32)
            - self.pixel_offset
        ) / float(W)
        grid_y, grid_x = torch.meshgrid(y, x)

        azimuth = grid_x * 2 * torch.pi - torch.pi - self.angle_offset
        if type(ir) != list and type(ir) != tuple:
            ir = [-ir, ir]
        if len(ir) == 2:
            grid_y = (
                grid_y
                * (
                    torch.arange(
                        H, 0, -1, device="cuda", dtype=torch.float32
                    ).unsqueeze(-1)
                    - self.pixel_offset
                )
                / float(H)
            )
            inclination = grid_y * (ir[1] - ir[0]) + ir[0]
        else:
            inclination = grid_y * torch.tensor(
                (ir), device="cuda", dtype=torch.float32
            ).flip(0).unsqueeze(-1)
        rays_x = torch.cos(inclination) * torch.cos(azimuth)
        rays_y = torch.cos(inclination) * torch.sin(azimuth)
        rays_z = torch.sin(inclination)

        rays_d = torch.stack([rays_x, rays_y, rays_z], dim=-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        points = rays_d * range_map[..., None].cuda()
        points = points @ sensor2world[:3, :3].T.cuda() + sensor2world[:3, 3].cuda()
        return points

    def get_range_rays(self, frame):
        ir = self.inclination_bounds
        sensor2world = self.sensor2world[frame]
        sensor_center = self.sensor_center[frame]

        rays_o = sensor_center.cuda()[None, None, ...].expand(self.H, self.W, 3)
        y = torch.ones(self.H, device="cuda", dtype=torch.float32)
        x = (
            torch.arange(self.W, 0, -1, device="cuda", dtype=torch.float32)
            - self.pixel_offset
        ) / float(self.W)
        grid_y, grid_x = torch.meshgrid(y, x)

        azimuth = grid_x * 2 * torch.pi - torch.pi - self.angle_offset
        if type(ir) != list and type(ir) != tuple:
            ir = [-ir, ir]
        if len(ir) == 2:
            grid_y = (
                grid_y
                * (
                    torch.arange(
                        self.H, 0, -1, device="cuda", dtype=torch.float32
                    ).unsqueeze(-1)
                    - self.pixel_offset
                )
                / float(self.H)
            )
            inclination = grid_y * (ir[1] - ir[0]) + ir[0]
        else:
            inclination = grid_y * torch.tensor(
                (ir), device="cuda", dtype=torch.float32
            ).flip(0).unsqueeze(-1)
        rays_x = torch.cos(inclination) * torch.cos(azimuth)
        rays_y = torch.cos(inclination) * torch.sin(azimuth)
        rays_z = torch.sin(inclination)

        rays_d = torch.stack([rays_x, rays_y, rays_z], dim=-1)
        rays_d = rays_d @ sensor2world[:3, :3].T.cuda()
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d
