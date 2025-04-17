import argparse

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch
from lib.utils.image_utils import color_mapping


def vis_pcd(pcd_path, point_size=3.0, colormap=matplotlib.colormaps["rainbow"]):
    pcd = o3d.io.read_point_cloud(pcd_path)

    z = torch.from_numpy(np.asarray(pcd.points)[:, 2])
    max_q = torch.quantile(z, 0.9)
    min_q = torch.quantile(z, 0.1)
    z = (z - min_q) / (max_q - min_q)
    z = z.clamp(0, 1)

    color = color_mapping(z, colormap)
    pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy().astype(np.float64))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([1.0, 1.0, 1.0])

    vis.run()
    vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pcd",
        type=str,
        required=True,
        help="Path to the point cloud file (extracted format by eval.py)",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=3.0,
        help="Size of the points in the visualization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vis_pcd(pcd_path=args.pcd, point_size=args.point_size)
