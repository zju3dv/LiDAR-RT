/**
 * @file trace_surfels.h
 * @author xbillowy
 * @brief 
 * @version 0.1
 * @date 2024-08-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once
#include <cstdio>
#include <tuple>
#include <string>
#include <torch/extension.h>

#include "optix_tracer/optix_wrapper.h"


void 
BuildAccelerationStructure(
    OptiXStateWrapper& stateWrapper,
    torch::Tensor& vertices,
    torch::Tensor& triangles,
    unsigned int rebuild);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
TraceSurfelsCUDA(
    const OptiXStateWrapper& stateWrapper,
    const bool training,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& vertices,
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& colors_precomp,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const float scale_modifier,
    const torch::Tensor& rotations,
    const torch::Tensor& transMat_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TraceSurfelsBackwardCUDA(
    const OptiXStateWrapper& stateWrapper,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& vertices,
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& colors_precomp,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const float scale_modifier,
    const torch::Tensor& rotations,
    const torch::Tensor& transMat_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug,
    const torch::Tensor& out_attr_float32,
    const torch::Tensor& out_attr_uint32,
    const torch::Tensor& dL_dout_attr_float32
);