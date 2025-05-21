import math
import os
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from lib.scene import BoundingBox, LiDARSensor
from lib.utils.console_utils import *

# from utils.kitti_utils import LiDAR_2_Pano_KITTI


def load_lidar2ego(base_dir, seq):
    cam2velo = np.asarray(
        [
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    ).reshape(4, 4)
    cam2ego = np.asarray(
        [
            0.0371783278,
            -0.0986182135,
            0.9944306009,
            1.5752681039,
            0.9992675562,
            -0.0053553387,
            -0.0378902567,
            0.0043914093,
            0.0090621821,
            0.9951109327,
            0.0983468786,
            -0.6500000000,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    ).reshape(4, 4)
    velo2ego = cam2ego @ np.linalg.inv(cam2velo)

    return velo2ego


def load_ego2world(file_path):
    ego2world = {}
    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.split()
        frame = int(parts[0])
        values = [float(x) for x in parts[1:]]
        matrix = np.array(values).reshape(3, 4)  # (3, 4)
        ego2world[frame] = matrix

    return ego2world


def load_lidar_point(lidar_dir, frames):
    lidar_points = {}
    for frame in range(frames[0], frames[1] + 1):
        with open(os.path.join(lidar_dir, f"{str(frame).zfill(10)}.bin"), "rb") as f:
            lidar_points[frame] = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    return lidar_points


def load_lidar_bbox(lidar_bbox_dir, full_seq, args, using_cache=True):
    bboxes: dict[str, BoundingBox] = {}
    bbox_pickle_dir = os.path.join(lidar_bbox_dir, "cache")
    os.makedirs(bbox_pickle_dir, exist_ok=True)
    bbox_pickle_path = os.path.join(bbox_pickle_dir, f"{full_seq}.pkl")

    # read pre cached pickle file
    if os.path.exists(bbox_pickle_path) and using_cache:
        try:
            with open(bbox_pickle_path, "rb") as fp:
                bboxes = pickle.load(fp)
            print("Read pickle file from: ", bbox_pickle_path)
            return bboxes
        except:
            print(red(f"Error: Cannot read pickle file from {bbox_pickle_path}"))

    xml_path = os.path.join(lidar_bbox_dir, full_seq + ".xml")
    with open(xml_path, "r") as f:
        xml_content = f.read()
    root = ET.fromstring(xml_content)

    for obj in root:
        metadata = {
            "label": obj.find("label").text,
            "instanceId": obj.find("instanceId").text,
            "category": obj.find("category").text,
            "timestamp": int(obj.find("timestamp").text),
            "dynamic": int(obj.find("dynamic").text),
            "transform": {
                "rows": int(obj.find("transform/rows").text),
                "cols": int(obj.find("transform/cols").text),
                "data": [float(val) for val in obj.find("transform/data").text.split()],
            },
        }
        object_type = metadata["label"]
        object_id = metadata["instanceId"]

        if (
            metadata["timestamp"] < args.frame_length[0]
            or metadata["timestamp"] > args.frame_length[1]
        ):
            continue
        if object_type not in ["car", "truck", "bus"]:
            continue

        transform = np.array(metadata["transform"]["data"]).reshape(
            metadata["transform"]["rows"], metadata["transform"]["cols"]
        )

        if object_id not in bboxes:
            U, S, V = np.linalg.svd(transform[:3, :3])
            bboxes[object_id] = BoundingBox(1, object_id, S)

        bboxes[object_id].add_frame_kitti(metadata["timestamp"], transform)

    if not os.path.exists(bbox_pickle_path):
        try:
            with open(bbox_pickle_path, "wb") as fp:
                pickle.dump(bboxes, fp)
            print(blue(f"Save pickle file to {bbox_pickle_path}"))
        except:
            os.remove(bbox_pickle_path)
            print(red(f"Error: Cannot save pickle file to {bbox_pickle_path}"))

    return bboxes


def load_SfM_clouds(SfM_clouds_dir):
    if os.path.exists(SfM_clouds_dir):
        xyzs, rgbs = [], []
        with open(SfM_clouds_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    xyzs.append(torch.tensor(tuple(map(float, elems[1:4]))))
                    rgbs.append(torch.tensor(tuple(map(int, elems[4:7]))))
        if xyzs == []:
            return None
        else:
            return [torch.stack(xyzs, dim=0), torch.stack(rgbs, dim=0)]
    else:
        return None


def load_kitti_raw(base_dir, args):
    if hasattr(args, "seq"):
        seq = args.seq
    else:
        seq = "0000"
    frames = args.frame_length
    full_seq = f"2013_05_28_drive_{seq}_sync"

    lidar_points = load_lidar_point(
        os.path.join(base_dir, "data_3d_raw", full_seq, "velodyne_points", "data"),
        frames,
    )
    lidar2ego = load_lidar2ego(base_dir, seq)
    ego2world = load_ego2world(
        os.path.join(base_dir, "data_pose", full_seq, "poses.txt")
    )

    W, H = 1030, 66
    inc_buttom, inc_top = math.radians(-24.9), math.radians(2.0)
    azimuth_left, azimuth_right = np.pi, -np.pi
    max_depth = 80.0
    h_res = (azimuth_right - azimuth_left) / W
    v_res = (inc_buttom - inc_top) / H

    lidar = LiDARSensor(
        sensor2ego=lidar2ego,
        name="velo",
        inclination_bounds=(inc_buttom, inc_top),
        data_type=args.data_type,
    )
    last_ego2world = None
    if frames[0] not in ego2world.keys():
        for pre_frame in range(frames[0] - 1, -1, -1):
            if pre_frame in ego2world.keys():
                last_ego2world = ego2world[pre_frame]
                break

    for frame in range(frames[0], frames[1] + 1):
        xyzs, intensities = lidar_points[frame][:, :3], lidar_points[frame][:, 3]
        x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
        dists = np.linalg.norm(lidar_points[frame][:, :3], axis=1)

        range_map = np.ones((H, W)) * -1
        intensity_map = np.ones((H, W)) * -1

        for xyz, intensity, dist in zip(xyzs, intensities, dists):
            x, y, z = xyz
            azimuth = np.arctan2(y, x)
            inclination = np.arctan2(z, np.sqrt(x**2 + y**2))

            if dist > max_depth:
                continue

            w_idx = np.round((azimuth - azimuth_left) / h_res).astype(int)
            h_idx = np.round((inclination - inc_top) / v_res).astype(int)

            if (w_idx < 0) or (w_idx >= W) or (h_idx < 0) or (h_idx >= H):
                continue

            if range_map[h_idx, w_idx] == -1:
                range_map[h_idx, w_idx] = dist
                intensity_map[h_idx, w_idx] = intensity
            elif range_map[h_idx, w_idx] > dist:
                range_map[h_idx, w_idx] = dist
                intensity_map[h_idx, w_idx] = intensity

        range_image_r1 = np.stack([range_map, intensity_map], axis=-1)
        range_image_r2 = np.ones_like(range_image_r1) * -1

        if frame in ego2world.keys():
            last_ego2world = ego2world[frame]
        range_image_r1[range_image_r1 == -1] = 0
        range_image_r2[range_image_r2 == -1] = 0
        lidar.add_frame(frame, last_ego2world, range_image_r1, range_image_r2)

    lidar_bbox = load_lidar_bbox(
        os.path.join(base_dir, "data_3d_bboxes", "train"),
        full_seq,
        args,
        using_cache=False,
    )

    return lidar, lidar_bbox
