// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <string>
#include <iterator>
#include <optix.h>
#include <optix_stubs.h>

#include "common.h"
#include "params.h"
#pragma comment  (lib, "Advapi32.lib")


// Define roundUp function
template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
  return ((x + y - 1) / y) * y;
}


// Typedef different SBT records
typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


struct OptiXState
{
    // OptiX context
    OptixDeviceContext context = nullptr;
    OptixTraversableHandle gas_handle;
    void *d_gas_output_buffer;

    // Forward module, SBT, pipeline
    OptixModule module_surfel_tracing_forward = nullptr;
    OptixShaderBindingTable sbt_surfel_tracing_forward = {};
    OptixPipeline pipelie_surfel_tracing_forward = nullptr;
    // Backward module, SBT, pipeline
    OptixModule module_surfel_tracing_backward = nullptr;
    OptixShaderBindingTable sbt_surfel_tracing_backward = {};
    OptixPipeline pipelie_surfel_tracing_backward = nullptr;
};


class OptiXStateWrapper
{
public:
    // OptiX state
    OptiXState* optixState;

    // Constructor and destructor
    OptiXStateWrapper(const std::string& pkg_dir);
    ~OptiXStateWrapper();
};
