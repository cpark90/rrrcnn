// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <torch/extension.h>
#include "group_conv/group_conv.h"
#include "hadamard_conv/hadamard_conv.h"
#include "hadamard_layer/hadamard_layer.h"
#include "skeleton_conv/skeleton_conv.h"
#include "continuous_conv/continuous_conv.h"
#include "deformed_feature_map_no_grad/deform_feature_map.h"
#include "deformed_orientation/deform_orientation.h"
#include "deformable_by_grad/deformable_by_grad.h"

namespace continuous {

#ifdef WITH_CUDA
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#ifdef WITH_CUDA
  std::ostringstream oss;

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else
  return std::string("not available");
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("group_conv_forward", &group_conv_forward, "group_conv_forward");
  m.def(
      "group_conv_backward_input",
      &group_conv_backward_input,
      "group_conv_backward_input");
  m.def(
      "group_conv_backward_filter",
      &group_conv_backward_filter,
      "group_conv_backward_filter");

  m.def("hadamard_conv_forward", &hadamard_conv_forward, "hadamard_conv_forward");
  m.def(
      "hadamard_conv_backward_input",
      &hadamard_conv_backward_input,
      "hadamard_conv_backward_input");
  m.def(
      "hadamard_conv_backward_filter",
      &hadamard_conv_backward_filter,
      "hadamard_conv_backward_filter");

  m.def("hadamard_layer_forward", &hadamard_layer_forward, "hadamard_layer_forward");
  m.def(
      "hadamard_layer_backward_input",
      &hadamard_layer_backward_input,
      "hadamard_layer_backward_input");

  m.def("deform_feature_map_forward", &deform_feature_map_forward, "deform_feature_map_forward");
  m.def(
      "deform_feature_map_backward_input",
      &deform_feature_map_backward_input,
      "deform_feature_map_backward_input");

  m.def("deformable_by_grad_forward", &deformable_by_grad_forward, "deformable_by_grad_forward");
  m.def(
      "deformable_by_grad_backward_input",
      &deformable_by_grad_backward_input,
      "deformable_by_grad_backward_input");

  m.def("deform_orientation_forward", &deform_orientation_forward, "deform_orientation_forward");
  m.def(
      "deform_orientation_backward_input",
      &deform_orientation_backward_input,
      "deform_orientation_backward_input");

  m.def("skeleton_conv_forward", &skeleton_conv_forward, "skeleton_conv_forward");
  m.def(
      "skeleton_conv_backward_input",
      &skeleton_conv_backward_input,
      "skeleton_conv_backward_input");
  m.def(
      "skeleton_conv_backward_filter",
      &skeleton_conv_backward_filter,
      "skeleton_conv_backward_filter");

  m.def("continuous_conv_forward", &continuous_conv_forward, "continuous_conv_forward");
  m.def(
      "continuous_conv_backward_input",
      &continuous_conv_backward_input,
      "continuous_conv_backward_input");
  m.def(
      "continuous_conv_backward_filter",
      &continuous_conv_backward_filter,
      "continuous_conv_backward_filter");
}
}
