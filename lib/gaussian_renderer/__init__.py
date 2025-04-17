import numpy as np
import torch
import torch.nn.functional as F
from diff_lidar_tracer import Tracer, TracingSettings
from lib.scene import Camera, GaussianModel, LiDARSensor
from lib.utils.general_utils import quaternion_raw_multiply
from lib.utils.graphics_utils import get_rays
from lib.utils.primitive_utils import primitiveTypeCallbacks
from lib.utils.sh_utils import eval_sh

tracer_2dgs = Tracer()
vertices, faces = None, None


def raytracing(
    frame: int,
    gaussian_assets: list[GaussianModel],
    sensor: LiDARSensor | Camera,
    background: torch.Tensor,
    args,
    scaling_modifier=1.0,
    override_color=None,
    decomp=False,
):

    if decomp == "background":
        gaussian_assets = gaussian_assets[:1]
    elif decomp == "object":
        gaussian_assets = gaussian_assets[1:]

    if isinstance(sensor, Camera):
        sensor_center = sensor.camera_center
        focal = 0.5 * sensor.image_width / np.tan(0.5 * sensor.FoVx)
        K = np.array(
            [
                [focal, 0, 0.5 * sensor.image_width],
                [0, focal, 0.5 * sensor.image_height],
                [0, 0, 1],
            ]
        )
        rays_o, rays_d = get_rays(K, sensor.world_view_transform.T.inverse()[:3, :4])
    elif isinstance(sensor, LiDARSensor):
        rays_o, rays_d = sensor.get_range_rays(frame)
        sensor_center = sensor.sensor_center[frame]
    elif isinstance(sensor, tuple):
        rays_o, rays_d = sensor[0], sensor[1]
        sensor_center = sensor[2]
    else:
        raise ValueError("sensor type not supported")

    tracer = tracer_2dgs
    primitiveCallback = primitiveTypeCallbacks["2DRectangle"]

    tracer_settings = TracingSettings(
        image_height=None,
        image_width=None,
        tanfovx=None,
        tanfovy=None,
        bg=background.cuda(),
        scale_modifier=1.0,
        viewmatrix=torch.Tensor([]).cuda(),
        projmatrix=torch.Tensor([]).cuda(),
        sh_degree=gaussian_assets[0].active_sh_degree,
        campos=sensor_center.cuda(),
        prefiltered=False,
        debug=False,
    )

    all_means3D = []
    all_opacities = []
    all_scales = []
    obj_rot, rot_in_local = [], []
    all_cov3D_precomp = []
    all_shs = []
    all_colors_precomp = []
    for pc in gaussian_assets[:]:
        means3D = pc.get_world_xyz(frame)
        opacity = pc.get_opacity
        all_means3D.append(means3D)
        all_opacities.append(opacity)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        if args.pipe.compute_cov3D_python:
            all_cov3D_precomp.append(pc.get_covariance(scaling_modifier, frame))
        else:
            all_scales.append(pc.get_scaling)
            # all_rotations.append(pc.get_rotation(timer1))
            r1, r2 = pc.get_rotation(frame)
            obj_rot.append(r1.expand(r2.shape[0], -1))
            rot_in_local.append(r2)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        if override_color is None:
            if args.pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 4, (pc.max_sh_degree + 1) ** 2
                )
                dir_pp = pc.get_world_xyz(frame) - sensor_center.repeat(
                    pc.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                all_colors_precomp.append(torch.clamp_min(sh2rgb + 0.5, 0.0))
            else:
                all_shs.append(pc.get_features)
        else:
            all_colors_precomp.append(override_color)

    # Concatenate all tensors
    means3D = torch.cat(all_means3D, dim=0)
    opacity = torch.cat(all_opacities, dim=0)
    cov3D_precomp = torch.cat(all_cov3D_precomp, dim=0) if all_cov3D_precomp else None
    scales = torch.cat(all_scales, dim=0) if all_scales else None

    if decomp == "background" or not args.dynamic:
        rotations = rot_in_local[0]
    elif decomp == "object":
        obj_rot = torch.cat(obj_rot, dim=0)  # exclude the background
        rot_in_local = torch.cat(rot_in_local, dim=0)
        rot_in_local = torch.nn.functional.normalize(rot_in_local, dim=1)
        rotations = quaternion_raw_multiply(None, obj_rot, rot_in_local)
    else:
        obj_rot = torch.cat(obj_rot[1:], dim=0)  # exclude the background
        rots_bkgd = rot_in_local[0]
        rot_in_local = torch.cat(rot_in_local[1:], dim=0)
        rot_in_local = torch.nn.functional.normalize(rot_in_local, dim=1)
        rotations = quaternion_raw_multiply(None, obj_rot, rot_in_local)
        rotations = torch.cat([rots_bkgd, rotations], dim=0)
    shs = torch.cat(all_shs, dim=0) if all_shs else None
    colors_precomp = (
        torch.cat(all_colors_precomp, dim=0) if all_colors_precomp else None
    )

    grads3D = torch.zeros_like(means3D, requires_grad=True)
    try:
        means3D.retain_grad()
    except:
        pass

    vertices, faces, mesh_normals = primitiveCallback(
        means3D, scales, rotations, opacity
    )  # (V, 3), (F, 3)
    tracer.build_acceleration_structure(vertices, faces, rebuild=True)

    rendered_tensor, accum_gaussian_weights = tracer(
        ray_o=rays_o,  # (H, W, 3)
        ray_d=rays_d,  # (H, W, 3)
        mesh_normals=mesh_normals,  # (V, 3)
        means3D=means3D,  # (P, 3)
        grads3D=grads3D,  # (P, 3)
        shs=shs,  # (P, 3, M)
        colors_precomp=None,
        opacities=opacity,  # (P, 1)
        scales=scales,  # (P, 3)
        rotations=rotations,  # (P, 4)
        cov3Ds_precomp=None,
        tracer_settings=tracer_settings,
    )

    # mean2D
    intensities = rendered_tensor[:, :, 0:1]
    rayhit_logits = rendered_tensor[:, :, 1:2]
    raydrop_logits = rendered_tensor[:, :, 2:3]
    depth = rendered_tensor[:, :, 3:4]

    if args.opt.use_rayhit:
        logits = torch.cat([rayhit_logits, raydrop_logits], dim=-1)
        prob = F.softmax(logits, dim=-1)
        raydrop_prob = prob[..., 1:2]
    else:
        raydrop_prob = torch.sigmoid(raydrop_logits)

    return {
        "depth": depth,
        "intensity": intensities,
        "raydrop": raydrop_prob,
        "means3D": means3D,
        "accum_gaussian_weight": accum_gaussian_weights.unsqueeze(-1),
    }
