/**
 * @file ext.cpp
 * @author xbillowy
 * @brief 
 * @version 0.1
 * @date 2024-08-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <torch/extension.h>

#include "trace_surfels.h"
#include "optix_tracer/optix_wrapper.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<OptiXStateWrapper>(m, "OptiXStateWrapper").def(pybind11::init<const std::string &>());
  m.def("build_acceleration_structure", &BuildAccelerationStructure);
  m.def("trace_surfels", &TraceSurfelsCUDA);
  m.def("trace_surfels_backward", &TraceSurfelsBackwardCUDA);
}
