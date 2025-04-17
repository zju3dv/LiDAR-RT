// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <optix.h>
#include <cuda_runtime.h>

#include "config.h"
#include "glm/glm.hpp"


// Define the data types

#ifndef UINT32_MAX
typedef unsigned int uint32_t;
#endif

#ifndef UINT64_MAX
typedef unsigned long int uint64_t;
#endif


// Define the data structure used in the OptiX kernel
struct RayGenData {};
struct HitGroupData {};
struct MissData {};


// Define the global data structure used in the OptiX kernel
struct Params
{
    // OptiX handler
    OptixTraversableHandle handle;

    // Global parameters
    bool training;  // training or testing

    // Input parameters
    int P, H, W, D, M;  // Gaussian number, height, width, SHs number
    float3* ray_o;  // (H, W, 3), ray origin
    float3* ray_d;  // (H, W, 3), ray direction
    float3* vertices;  // (P * 2, 3), primitive vertices
    float* background;  // (3), background color
    glm::vec3* means3D;  // (P, 3), center coordinates
    float* shs;  // (P, M), SHs
    float* colors_precomp;  // (P, C), precomputed parameters
    float* opacities;  // (P, 1), opacities
    glm::vec2* scales;  // (P, 2), scales
    float scale_modifier;
    glm::vec4* rotations;  // (P, 4), rotations
    float* transMat_precomp;
    float* viewmatrix;
    float* projmatrix;
    glm::vec3* campos;

    // Output forward results
    float* out_attr_float32;  // (H, W, C), RGB color or other features
    int* out_attr_uint32;
    float* accum_gaussian_weights;
    
    // Input upstream gradients
    float* dL_dout_attr_float32;  // (H, W, C), gradient of RGB color or other features

    // Output gradients
    glm::vec3* dL_dmeans3D;  // (P, 3), gradient of center coordinates
    glm::vec3* dL_dgrads3D_abs;
    glm::vec3* dL_dshs;  // (P, M, 3), gradient of SHs
    float* dL_dcolors;  // (P, C), gradient of middle colors
    float* dL_dopacities;  // (P, 1), gradient of opacities
    glm::vec2* dL_dscales;  // (P, 2), gradient of scales
    glm::vec4* dL_drotations;  // (P, 4), gradient of rotations
    float* dL_dtransMat_precomp;  // (P, 9), gradient of trans matrix
};


// Define the primitive info
struct IntersectionInfo {
    float tmx;  // t range along the ray
    uint32_t idx;  // intersection primitive ID
};
// Typedef
typedef struct IntersectionInfo IntersectionInfo;

// Define the ray payload.
// Ray pyaload is used to pass data between optixTrace
// and the programs invoked during ray traversal.
struct RayPayload {
    float dpt;  // trace depth during the whole chunkify tracing
    uint32_t cnt;  // record number of intersections for one chunk
    IntersectionInfo* buffer;  // hit buffer for one chunk
};
// Typedef
typedef struct RayPayload RayPayload;
