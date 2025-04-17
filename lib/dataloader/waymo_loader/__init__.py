import os
import struct
from typing import Dict

import tensorflow as tf
import torch
from tqdm import tqdm

tf.config.set_visible_devices([], "GPU")

from lib.dataloader.waymo_loader.waymo_protobuf import dataset_pb2
from lib.scene.bounding_box import BoundingBox
from lib.scene.lidar_sensor import LiDARSensor


def decompress_range_image(compressed):
    """
    Add two numbers and return the result.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    decompress_str = tf.io.decode_compressed(compressed, "ZLIB")
    decompress_data = dataset_pb2.MatrixFloat()
    decompress_data.ParseFromString(bytearray(decompress_str.numpy()))
    range_image_tensor = torch.tensor(
        decompress_data.data, dtype=torch.float32
    ).reshape(tuple(decompress_data.shape.dims))
    return range_image_tensor


def load_waymo_raw(base_dir, args):
    for filename in os.listdir(base_dir):
        if filename.endswith(".tfrecord"):
            fp = os.path.join(base_dir, filename)

    dataset = tf.data.TFRecordDataset(fp, compression_type="")
    dataset = list(dataset)
    lidar: LiDARSensor = None
    bboxes: Dict[str, BoundingBox] = {}  # frame * n

    pbar = tqdm(total=(args.frame_length[1] + 1 - args.frame_length[0]))
    for frame in range(args.frame_length[0], args.frame_length[1] + 1):
        record = dataset[frame]
        frame_data = dataset_pb2.Frame()
        frame_data.ParseFromString(bytearray(record.numpy()))
        for i in range(5):
            name = frame_data.context.laser_calibrations[i].name
            if name != 1:
                continue

            if lidar is None:
                lidar2ego = frame_data.context.laser_calibrations[i].extrinsic.transform
                lidar2ego = torch.tensor(lidar2ego, dtype=torch.float32).reshape(4, 4)
                if frame_data.context.laser_calibrations[i].beam_inclinations:
                    beam_inclination = list(
                        frame_data.context.laser_calibrations[i].beam_inclinations
                    )
                else:
                    beam_inclination_min = frame_data.context.laser_calibrations[
                        i
                    ].beam_inclination_min
                    beam_inclination_max = frame_data.context.laser_calibrations[
                        i
                    ].beam_inclination_max
                    beam_inclination = [beam_inclination_min, beam_inclination_max]
                lidar = LiDARSensor(
                    name=name,
                    sensor2ego=lidar2ego,
                    inclination_bounds=beam_inclination,
                    data_type=args.data_type,
                )

            ego2world = torch.tensor(
                frame_data.pose.transform, dtype=torch.float32
            ).reshape(4, 4)

            decompressed_dir = f"{base_dir}/cache"
            os.makedirs(decompressed_dir, exist_ok=True)
            decompressed_path = os.path.join(
                decompressed_dir, f"decompressed_frame_{frame}_sensor_{name}.pt"
            )
            if os.path.exists(decompressed_path):
                range_image_r1, range_image_r2 = torch.load(decompressed_path)
            else:
                for lidar_data in frame_data.lasers:
                    if lidar_data.name == name:
                        range_image_r1 = decompress_range_image(
                            lidar_data.ri_return1.range_image_compressed
                        )
                        range_image_r2 = decompress_range_image(
                            lidar_data.ri_return2.range_image_compressed
                        )
                        range_image_r1[..., 1] = torch.clamp(
                            range_image_r1[..., 1], max=1
                        )
                        range_image_r1[..., 0:2][range_image_r1[..., 0:2] == -1] = 0
                torch.save((range_image_r1, range_image_r2), decompressed_path)

            lidar.add_frame(
                frame=frame, ego2world=ego2world, r1=range_image_r1, r2=range_image_r2
            )

        for labels in frame_data.laser_labels:
            box = labels.box
            id, tp = labels.id, labels.type
            x, y, z, l, w, h, yaw = (
                box.center_x,
                box.center_y,
                box.center_z,
                box.length,
                box.width,
                box.height,
                box.heading,
            )
            metadata = [id, x, y, z, l, w, h, yaw, tp]
            if id not in bboxes:
                object_type = int(metadata[8])
                object_id = metadata[0]
                float_data = [float(x) for x in metadata[4:7]]
                size = torch.tensor(float_data).float().cuda()
                bboxes[id] = BoundingBox(object_type, object_id, size)
            bboxes[id].add_frame_waymo(frame, metadata, ego2world)

        pbar.update(1)
    pbar.close()
    return lidar, bboxes
