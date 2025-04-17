import argparse
import json
import os

import cv2
import imageio
import lpips
import numpy as np
import open3d as o3d
import torch
from lib import dataloader
from lib.arguments import Args, parse
from lib.gaussian_renderer import raytracing
from lib.scene.unet import UNet
from lib.utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from lib.utils.console_utils import *
from lib.utils.image_utils import color_mapping
from skimage.metrics import structural_similarity
from tqdm import tqdm

colormap_ = 20


class LiDARRTMeter:
    def __init__(self, args) -> None:
        self.args = args
        scene = dataloader.load_scene(self.args.source_dir, args, test=False)
        gaussians = scene.gaussians_assets
        if not self.args.model_path:
            model_dir = scene.model_save_dir
            model_files = [
                file for file in os.listdir(model_dir) if file.endswith(".pth")
            ]
            for file in model_files:
                if "_good" in file:
                    self.args.model_path = os.path.join(model_dir, file)
                    break

        if not self.args.model_path:
            raise ValueError("Model path is not provided. No good model found.")

        model_params, first_iter = torch.load(self.args.model_path)
        print("Loading model with iteration: ", first_iter)
        scene.restore(model_params, self.args.opt)

        self.unet = None
        if args.unet:
            in_channels = 9 if args.refine.use_spatial else 3
            self.unet = UNet(in_channels=in_channels, out_channels=1).cuda()
            unet_params = torch.load(args.unet)
            self.unet.load_state_dict(unet_params)
        self.gaussians = gaussians
        self.scene = scene
        self.sensor = "lidar"
        self.eval_type = self.args.eval_type
        self.eval_frames = self.args.eval_frames
        self.train_frames = [
            frame_id
            for frame_id in range(
                self.args.frame_length[0], self.args.frame_length[1] + 1
            )
            if frame_id not in self.args.eval_frames
        ]

        #! eval settings
        # bg_color = [1, 1, 1] if model_cfg.white_background else [0, 0, 0]
        self.background = torch.tensor(
            [0, 0, 1], device="cuda"
        ).float()  # intensity, hit prob, drop prob
        self.scale = 1.0
        self.intensity_scale = 1.0
        self.raydrop_ratio = 0.4
        self.colormap = cv2.COLORMAP_JET

        self.lpips_fn = lpips.LPIPS(net="alex").eval()

        self.save_eval = self.args.save_eval
        self.save_image = self.args.save_image
        self.save_pcd = self.args.save_pcd
        self.use_gt_mask = self.args.use_gt_mask

        # self.save_path = self.args.save_path
        if args.save_path:
            self.save_path = args.save_path
        else:
            # get the parent dir of model path
            model_dir = os.path.dirname(args.model_path)
            parent_dir = os.path.dirname(model_dir)
            self.save_path = os.path.join(parent_dir, f"evals/{first_iter}")
            print("Save eval results to: ", blue(self.save_path))
        self.video_dir = os.path.join(self.save_path, self.eval_type, "videos")
        self.image_dir = os.path.join(self.save_path, self.eval_type, "images")
        self.pcd_dir = os.path.join(self.save_path, self.eval_type, "pointclouds")
        self.eval_dir = os.path.join(self.save_path, self.eval_type, "metrics")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

    def record_render(self, gaussians, background, scene, frame_id, args):
        render_dict = dict()

        points, _ = scene.train_lidar.inverse_projection(frame_id)
        points = torch.cat([points, torch.ones((points.shape[0], 1))], dim=1).cuda()
        if self.sensor == "camera":
            sensor = scene.train_lidar.gen_norot_cam(frame_id).cuda()
            points_camera = points.float().cuda() @ sensor.world_view_transform
            points_proj = points_camera @ sensor.projection_matrix
            points_proj = points_proj[:, :3] / points_proj[:, 3, None]
            uvz = torch.zeros_like(points).cuda().float()
            uvz[:, 0] = ((points_proj[:, 0] + 1.0) * sensor.image_width - 1) * 0.5
            uvz[:, 1] = ((points_proj[:, 1] + 1.0) * sensor.image_height - 1) * 0.5
            uvz[:, 2] = points_camera[:, 2]
            mask = (
                (uvz[:, 2] > 0)
                & (uvz[:, 1] >= 0)
                & (uvz[:, 1] < sensor.image_height)
                & (uvz[:, 0] >= 0)
                & (uvz[:, 0] < sensor.image_width)
            )
            uvz = uvz[mask]
        else:
            sensor = scene.train_lidar
        rendered_pkg = raytracing(frame_id, gaussians, sensor, background, args)
        rendered_depth = rendered_pkg["depth"].detach()
        rendered_intensity = rendered_pkg["intensity"].detach()
        rendered_raydrop = rendered_pkg["raydrop"].detach()

        if self.unet:
            H, W = rendered_depth.shape[0], rendered_depth.shape[1]
            input_depth = rendered_depth.reshape(1, H, W)
            input_intensity = rendered_intensity.reshape(1, H, W)
            input_raydrop = rendered_raydrop.reshape(1, H, W)
            raydrop_prob = torch.cat(
                [input_raydrop, input_intensity, input_depth], dim=0
            )
            if args.refine.use_spatial:
                ray_o, ray_d = scene.train_lidar.get_range_rays(frame_id)
                raydrop_prob = torch.cat(
                    [raydrop_prob, ray_o.permute(2, 0, 1), ray_d.permute(2, 0, 1)],
                    dim=0,
                )
            raydrop_prob = raydrop_prob.unsqueeze(0)
            rendered_raydrop = self.unet(raydrop_prob).detach().reshape(H, W, 1)

        gt_rayhit = scene.train_lidar.get_mask(frame_id).unsqueeze(-1)
        if self.save_image:
            gt_rayhit_vis = gt_rayhit.repeat(1, 1, 3)
            gt_mask_vis = gt_rayhit_vis.cpu().numpy()
            gt_rayhit_vis = np.uint8(gt_mask_vis * 255)
        gt_rayhit = gt_rayhit.cpu().numpy()

        gt_depth = scene.train_lidar.get_depth(frame_id)
        if self.save_image:
            gt_depth_vis = (gt_depth - gt_depth.min()) / (
                gt_depth.max() - gt_depth.min()
            )
            gt_depth_vis = gt_depth_vis.cpu().numpy()
            gt_depth_vis = (
                color_mapping(gt_depth_vis, colormap_)[..., :3] * 255
            ).astype(np.uint8)
            gt_depth_vis = gt_depth_vis * gt_mask_vis
        gt_depth = gt_depth.unsqueeze(-1).cpu().numpy()

        gt_intensity = scene.train_lidar.get_intensity(frame_id)
        gt_intensity = gt_intensity.clamp(0, 1)
        if self.save_image:
            gt_intensity_vis = (gt_intensity - gt_intensity.min()) / (
                gt_intensity.max() - gt_intensity.min()
            )
            gt_intensity_vis = gt_intensity_vis.cpu().numpy()
            gt_intensity_vis = (
                color_mapping(gt_intensity_vis, colormap_)[..., :3] * 255
            ).astype(np.uint8)
            gt_intensity_vis = gt_intensity_vis * gt_mask_vis
        gt_intensity = gt_intensity.unsqueeze(-1).cpu().numpy()

        rendered_rayhit = rendered_raydrop < self.raydrop_ratio
        # rendered_rayhit = rendered_rayhit & (rendered_depth < scene.train_lidar.max_depth)
        if self.save_image:
            rendered_rayhit_vis = rendered_rayhit.repeat(1, 1, 3).cpu().numpy()
            rendered_rayhit_vis = np.uint8(rendered_rayhit_vis * 255)
        rendered_rayhit = rendered_rayhit.cpu().numpy()
        mask = gt_rayhit if self.use_gt_mask else rendered_rayhit

        gt_pts = scene.train_lidar.inverse_projection_with_range(
            frame_id, gt_depth, gt_rayhit
        )
        gt_pts = gt_pts.cpu().numpy().astype(np.float64)
        rendered_pts = scene.train_lidar.inverse_projection_with_range(
            frame_id, rendered_depth, mask
        )
        rendered_pts = rendered_pts.cpu().numpy().astype(np.float64)

        if self.save_pcd:
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)
            gt_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(gt_pts) * 0.9)

            rendered_pcd = o3d.geometry.PointCloud()
            rendered_pcd.points = o3d.utility.Vector3dVector(rendered_pts)
            rendered_pcd.colors = o3d.utility.Vector3dVector(
                np.ones_like(rendered_pts) * 0.9
            )

        rendered_depth = rendered_depth.cpu().numpy()
        if self.save_image:
            nonzero_mask = rendered_depth != 0
            rendered_depth_vis = (rendered_depth - gt_depth.min()) / (
                gt_depth.max() - gt_depth.min()
            )
            rendered_depth_vis = (
                color_mapping(rendered_depth_vis[..., 0], colormap_) * 255
            ).astype(np.uint8)
            rendered_depth_vis = rendered_depth_vis * (mask & nonzero_mask)
        rendered_depth = rendered_depth * mask

        rendered_intensity = rendered_intensity.clamp(0, 1.0).cpu().numpy()
        if self.save_image:
            nonzero_mask = rendered_intensity != 0
            rendered_intensity_vis = rendered_intensity
            rendered_intensity_vis = (rendered_intensity_vis - gt_intensity.min()) / (
                gt_intensity.max() - gt_intensity.min()
            )
            rendered_intensity_vis = (
                color_mapping(rendered_intensity_vis[..., 0], colormap_) * 255
            ).astype(np.uint8)
            rendered_intensity_vis = rendered_intensity_vis * (mask & nonzero_mask)
        rendered_intensity = rendered_intensity * mask

        render_dict.update(
            {
                "rendered_depth": rendered_depth,
                "rendered_intensity": rendered_intensity,
                "rendered_rayhit": rendered_rayhit,
                "gt_depth": gt_depth,
                "gt_intensity": gt_intensity,
                "gt_rayhit": gt_rayhit,
                "gt_pts": gt_pts,
                "rendered_pts": rendered_pts,
            }
        )

        if self.save_image:
            render_dict.update(
                {
                    "rendered_depth_vis": rendered_depth_vis,
                    "rendered_intensity_vis": rendered_intensity_vis,
                    "rendered_rayhit_vis": rendered_rayhit_vis,
                    "gt_depth_vis": gt_depth_vis,
                    "gt_intensity_vis": gt_intensity_vis,
                    "gt_rayhit_vis": gt_rayhit_vis,
                }
            )

        if self.save_pcd:
            render_dict.update(
                {
                    "gt_pcd": gt_pcd,
                    "rendered_pcd": rendered_pcd,
                }
            )

        return render_dict

    def compute_fscore(self, dist1, dist2, threshold=0.001):
        """
        Calculates the F-score between two point clouds with the corresponding threshold value.
        :param dist1: Batch, N-Points
        :param dist2: Batch, N-Points
        :param th: float
        :return: fscore, precision, recall
        """
        # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean
        # distances, so you should adapt the threshold accordingly.
        precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
        precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
        fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
        fscore[torch.isnan(fscore)] = 0
        return [fscore, precision_1, precision_2]

    def compute_depth_metrics(self, gt, pred, min_depth=1e-6, max_depth=80):
        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        mae = np.mean(np.abs(gt - pred))
        medae = np.median(np.abs(gt - pred))

        psnr_loss = 10 * np.log10(max_depth**2 / np.mean((pred - gt) ** 2))

        ssim_loss = structural_similarity(
            pred.squeeze(-1), gt.squeeze(-1), data_range=np.max(gt) - np.min(gt)
        )

        lpips_loss = self.lpips_fn(
            torch.from_numpy(pred).permute(2, 0, 1),
            torch.from_numpy(gt).permute(2, 0, 1),
            normalize=True,
        ).item()

        return [rmse, mae, medae, lpips_loss, ssim_loss, psnr_loss]

    def compute_intensity_metrics(
        self, gt, pred, min_intensity=1e-6, max_intensity=1.0
    ):
        pred[pred < min_intensity] = min_intensity
        pred[pred > max_intensity] = max_intensity
        gt[gt < min_intensity] = min_intensity
        gt[gt > max_intensity] = max_intensity

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        mae = np.mean(np.abs(gt - pred))
        medae = np.median(np.abs(gt - pred))

        psnr_loss = 10 * np.log10(max_intensity**2 / np.mean((pred - gt) ** 2))

        ssim_loss = structural_similarity(
            pred.squeeze(-1), gt.squeeze(-1), data_range=np.max(gt) - np.min(gt)
        )

        lpips_loss = self.lpips_fn(
            torch.from_numpy(pred).permute(2, 0, 1),
            torch.from_numpy(gt).permute(2, 0, 1),
            normalize=True,
        ).item()

        return [rmse, mae, medae, lpips_loss, ssim_loss, psnr_loss]

    def compute_raydrop_metrics(self, gt, pred):
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        preds_mask = np.where(pred > self.raydrop_ratio, 1, 0)
        acc = (preds_mask == gt).mean()

        TP = np.sum((gt == 1) & (preds_mask == 1))
        FP = np.sum((gt == 0) & (preds_mask == 1))
        TN = np.sum((gt == 0) & (preds_mask == 0))
        FN = np.sum((gt == 1) & (preds_mask == 0))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        return [rmse, acc, f1]

    def compute_points_metrics(self, gt, pred):
        chamLoss = chamfer_3DDist()
        dist1, dist2, idx1, idx2 = chamLoss(
            torch.FloatTensor(gt[None, ...]).cuda(),
            torch.FloatTensor(pred[None, ...]).cuda(),
        )
        chamfer_dis = dist1.mean() + dist2.mean()
        chamfer_dis = chamfer_dis.cpu()
        f_score, precision, recall = self.compute_fscore(dist1, dist2, threshold=0.05)
        f_score = f_score.cpu()[0]

        return [chamfer_dis, f_score]

    def run(self):
        frames = []
        eval_dict_all = dict()
        eval_depth = []
        eval_intensity = []
        eval_raydrop = []
        eval_points = []
        depth_metrics = ["rmse", "mae", "medae", "lpips_loss", "ssim", "psnr"]
        intensity_metrics = ["rmse", "mae", "medae", "lpips_loss", "ssim", "psnr"]
        raydrop_metrics = ["rmse", "acc", "f1"]
        points_metrics = ["chamfer_dist", "fscore"]
        depth_dict = dict()
        intensity_dict = dict()
        raydrop_dict = dict()
        points_dict = dict()

        eval_all_frame_dict = dict()

        if self.eval_type == "train":
            all_frames = self.train_frames
        elif self.eval_type == "test":
            all_frames = self.eval_frames
        elif self.eval_type == "all":
            all_frames = list(
                range(self.args.frame_length[0], self.args.frame_length[1] + 1)
            )
        else:
            raise ValueError("Invalid evaluation type.")

        for frame_id in tqdm(
            all_frames, total=len(all_frames), desc="Metric evaluation progress"
        ):
            eval_per_frame_dict = dict()
            depth_per_frame_dict = dict()
            intensity_per_frame_dict = dict()
            raydrop_per_frame_dict = dict()
            points_per_frame_dict = dict()

            render_dict = self.record_render(
                self.gaussians, self.background, self.scene, frame_id, self.args
            )

            depth_per_frame = self.compute_depth_metrics(
                render_dict["gt_depth"], render_dict["rendered_depth"]
            )
            intensity_per_frame = self.compute_intensity_metrics(
                render_dict["gt_intensity"], render_dict["rendered_intensity"]
            )
            raydrop_per_frame = self.compute_raydrop_metrics(
                1 - render_dict["gt_rayhit"], 1 - render_dict["rendered_rayhit"]
            )
            points_per_frame = self.compute_points_metrics(
                render_dict["gt_pts"], render_dict["rendered_pts"]
            )

            eval_depth.append(depth_per_frame)
            eval_intensity.append(intensity_per_frame)
            eval_raydrop.append(raydrop_per_frame)
            eval_points.append(points_per_frame)

            for metric, result in zip(depth_metrics, depth_per_frame):
                depth_per_frame_dict.update({metric: torch.tensor(result).cpu().item()})

            for metric, result in zip(intensity_metrics, intensity_per_frame):
                intensity_per_frame_dict.update(
                    {metric: torch.tensor(result).cpu().item()}
                )

            for metric, result in zip(raydrop_metrics, raydrop_per_frame):
                raydrop_per_frame_dict.update(
                    {metric: torch.tensor(result).cpu().item()}
                )

            for metric, result in zip(points_metrics, points_per_frame):
                points_per_frame_dict.update(
                    {metric: torch.tensor(result).cpu().item()}
                )

            eval_per_frame_dict.update(
                {
                    "depth": depth_per_frame_dict,
                    "intensity": intensity_per_frame_dict,
                    "raydrop": raydrop_per_frame_dict,
                    "points": points_per_frame_dict,
                }
            )

            eval_all_frame_dict.update({frame_id: eval_per_frame_dict})

            if self.save_image:
                concat_image = np.concatenate(
                    (
                        render_dict["rendered_depth_vis"],
                        render_dict["gt_depth_vis"],
                        # render_dict['rendered_depth_vis']-render_dict['gt_depth_vis'],
                        render_dict["rendered_intensity_vis"],
                        render_dict["gt_intensity_vis"],
                        # render_dict['rendered_intensity_vis']-render_dict['gt_intensity_vis'],
                        render_dict["rendered_rayhit_vis"],
                        render_dict["gt_rayhit_vis"],
                        # render_dict['rendered_rayhit_vis']-render_dict['gt_rayhit_vis'],
                    ),
                    axis=0,
                )
                rgb_image = cv2.cvtColor(concat_image, cv2.COLOR_BGR2RGB)
                image_path = os.path.join(
                    self.image_dir, f"{frame_id:4d}_concat_image.jpg"
                )
                imageio.imwrite(image_path, concat_image)
                frames.append(rgb_image)

            if self.save_pcd:
                o3d.io.write_point_cloud(
                    os.path.join(self.pcd_dir, f"{frame_id:04d}_gt_pcd.ply"),
                    render_dict["gt_pcd"],
                )
                o3d.io.write_point_cloud(
                    os.path.join(self.pcd_dir, f"{frame_id:04d}_rendered_pcd.ply"),
                    render_dict["rendered_pcd"],
                )

        eval_depth = np.mean(np.array(eval_depth), axis=0)
        eval_intensity = np.mean(np.array(eval_intensity), axis=0)
        eval_raydrop = np.mean(np.array(eval_raydrop), axis=0)
        eval_points = np.mean(np.array(eval_points), axis=0)

        for metric, result in zip(depth_metrics, eval_depth):
            depth_dict.update({metric: torch.tensor(result).mean().cpu().item()})

        for metric, result in zip(intensity_metrics, eval_intensity):
            intensity_dict.update({metric: torch.tensor(result).mean().cpu().item()})

        for metric, result in zip(raydrop_metrics, eval_raydrop):
            raydrop_dict.update({metric: torch.tensor(result).mean().cpu().item()})

        for metric, result in zip(points_metrics, eval_points):
            points_dict.update({metric: torch.tensor(result).mean().cpu().item()})

        eval_dict_all.update(
            {
                "depth": depth_dict,
                "intensity": intensity_dict,
                "raydrop": raydrop_dict,
                "points": points_dict,
            }
        )

        if self.save_eval:
            eval_path = os.path.join(self.eval_dir, "results_all.json")
            eval_per_frame_path = os.path.join(self.eval_dir, "results_per_frame.json")
            with open(eval_path, "w") as fp:
                json.dump(eval_dict_all, fp, indent=4)
            with open(eval_per_frame_path, "w") as fp:
                json.dump(eval_all_frame_dict, fp, indent=4)

        if self.save_image:
            video_path = os.path.join(self.video_dir, "render_video.mp4")
            imageio.mimsave(video_path, np.array(frames), fps=5)


def parse_args():
    parser = argparse.ArgumentParser(description="launch args")
    parser.add_argument("-dc", "--data_config_path", type=str, help="config path")
    parser.add_argument("-ec", "--exp_config_path", type=str, help="config path")
    parser.add_argument("-m", "--model", type=str, help="model path")
    parser.add_argument("-un", "--unet", type=str, help="unet path")
    parser.add_argument("-s", "--save", type=str, default="", help="save path")
    parser.add_argument(
        "-t", "--type", type=str, default="test", help="data involved (train/test/all)"
    )
    parser.add_argument(
        "-e", "--save_eval", action="store_true", help="save metrics or not"
    )
    parser.add_argument(
        "-i", "--save_image", action="store_true", help="save image or not"
    )
    parser.add_argument("-p", "--save_pcd", action="store_true", help="save pcd or not")
    parser.add_argument(
        "-u", "--use_gt_mask", action="store_true", help="use gt mask or not"
    )

    launch_args = parser.parse_args()
    args = parse(launch_args.exp_config_path)
    args = parse(launch_args.data_config_path, args)
    args.model_path = launch_args.model
    args.unet = launch_args.unet
    args.save_path = launch_args.save
    args.eval_type = launch_args.type
    args.save_eval = launch_args.save_eval
    args.save_image = launch_args.save_image
    args.save_pcd = launch_args.save_pcd
    args.use_gt_mask = launch_args.use_gt_mask

    return args


if __name__ == "__main__":
    args = parse_args()
    meter = LiDARRTMeter(args)
    meter.run()
