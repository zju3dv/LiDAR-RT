import sysconfig
import torch
import torch.nn as nn
from typing import NamedTuple

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class _Tracer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                optix_context,
                training,
                ray_o,
                ray_d,
                vertices,
                means3D,
                grads3D,
                shs,
                colors_precomp,
                opacities,
                scales,
                rotations,
                cov3Ds_precomp,
                tracer_settings,
                ):

        # Restructure arguments the way that the C++ lib expects them
        args = (optix_context,
                training,
                ray_o,
                ray_d,
                vertices,
                tracer_settings.bg,
                means3D,
                shs,
                tracer_settings.sh_degree,
                colors_precomp,
                opacities,
                scales,
                tracer_settings.scale_modifier,
                rotations,
                cov3Ds_precomp,
                tracer_settings.viewmatrix,
                tracer_settings.projmatrix,
                tracer_settings.campos,
                tracer_settings.prefiltered,
                tracer_settings.debug)

        # Invoke C++/CUDA/OptiX tracer
        if tracer_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                out_attr_float32, out_attr_uint32 = _C.trace_surfels(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            out_attr_float32, out_attr_uint32, accum_gaussian_weights = _C.trace_surfels(*args)

        # Keep relevant tensors for backward
        ctx.tracer_settings = tracer_settings
        ctx.optix_context = optix_context
        ctx.save_for_backward(ray_o, ray_d, vertices, means3D, shs, colors_precomp, opacities, scales, rotations, cov3Ds_precomp,
                              out_attr_float32, out_attr_uint32)

        # Return the per-Gaussian hit counter for training gradient filtering
        # acc_wet = a_weights if a_weights is not None else torch.zeros_like(means3D[:, 0], dtype=means3D.dtype, device=means3D.device)
        return out_attr_float32, accum_gaussian_weights

    @staticmethod
    def backward(ctx, grad_out_attr_float32, _):

        # Restore necessary values from context
        tracer_settings = ctx.tracer_settings
        optix_context = ctx.optix_context
        ray_o, ray_d, vertices, means3D, shs, colors_precomp, opacities, scales, rotations, cov3Ds_precomp, \
            out_attr_float32, out_attr_uint32 = ctx.saved_tensors
        # Restructure args as C++ method expects them
        args = (optix_context,
                ray_o,
                ray_d,
                vertices,
                tracer_settings.bg,
                means3D,
                shs,
                tracer_settings.sh_degree,
                colors_precomp,
                opacities,
                scales,
                tracer_settings.scale_modifier,
                rotations,
                cov3Ds_precomp,
                tracer_settings.viewmatrix,
                tracer_settings.projmatrix,
                tracer_settings.campos,
                tracer_settings.prefiltered,
                tracer_settings.debug,
                out_attr_float32,
                out_attr_uint32,
                grad_out_attr_float32)

        # Compute gradients for relevant tensors by invoking backward method
        if tracer_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means3D, grad_shs, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, grad_grads3D = _C.trace_surfels_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means3D, grad_shs, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, grad_grads3D = _C.trace_surfels_backward(*args)
        grads = (
            None,
            None,
            None,
            None,
            None,
            grad_means3D,
            grad_grads3D,
            grad_shs,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class TracingSettings(NamedTuple):
    image_height: int  # no use, only for compatibility
    image_width: int  # no use, only for compatibility
    tanfovx: float  # no use, only for compatibility
    tanfovy: float  # no use, only for compatibility
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class Tracer(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

        # Find the OptiX shared library
        self.pkg_dir = sysconfig.get_path('purelib') + '/diff_lidar_tracer'

        # Create the OptiX context
        self.optix_context = _C.OptiXStateWrapper(self.pkg_dir)

    def build_acceleration_structure(self,
                                     vertices: torch.Tensor,
                                     triangles: torch.Tensor,
                                     rebuild: bool = 1,
                                     ):
        # Invoke C++/CUDA/OptiX acceleration structure building routine
        self.vertices = vertices
        return _C.build_acceleration_structure(self.optix_context, vertices, triangles, rebuild)

    def forward(self,
                ray_o: torch.Tensor,
                ray_d: torch.Tensor,
                mesh_normals: torch.Tensor,
                means3D: torch.Tensor,
                grads3D: torch.Tensor,
                shs: torch.Tensor = None,
                colors_precomp: torch.Tensor = None,
                opacities: torch.Tensor = None,
                scales: torch.Tensor = None,
                rotations: torch.Tensor = None,
                cov3Ds_precomp: torch.Tensor = None,
                tracer_settings: TracingSettings = None,
                ):

        # Check if colors or SHs are provided
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        # Check if scales/rotations or cov3Ds_precomp is provided
        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or ((scales is not None or rotations is not None) and cov3Ds_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # Create dummy color tracing input
        if shs is None: shs = torch.Tensor([]).cuda()
        if colors_precomp is None: colors_precomp = torch.Tensor([]).cuda()
        # Create dummy tracing transformation matrix input
        if scales is None: scales = torch.Tensor([]).cuda()
        if rotations is None: rotations = torch.Tensor([]).cuda()
        if cov3Ds_precomp is None: cov3Ds_precomp = torch.Tensor([]).cuda()

        # Invoke the autograd function
        return _Tracer.apply(
            self.optix_context,
            self.training,
            ray_o,
            ray_d,
            self.vertices,
            means3D,
            grads3D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            tracer_settings
        )
