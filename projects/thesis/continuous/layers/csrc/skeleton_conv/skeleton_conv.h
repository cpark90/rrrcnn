// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace continuous {

#ifdef WITH_CUDA
int skeleton_conv_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    at::Tensor columns,
    int kN,
    float dilation,
    int step,
    int im2col_step);

int skeleton_conv_backward_input_cuda(
    at::Tensor input,
    at::Tensor output,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor weight,
    at::Tensor columns,
    int kN,
    float dilation,
    int step,
    int im2col_step);

int skeleton_conv_backward_parameters_cuda(
    at::Tensor input,
    at::Tensor output,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    int kN,
    float dilation,
    int step,
    float scale,
    int im2col_step);

#endif

inline int skeleton_conv_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    at::Tensor columns,
    int kN,
    float dilation,
    int step,
    int im2col_step) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(weight.type().is_cuda(), "weight tensor is not on GPU!");
    return skeleton_conv_forward_cuda(
        input,
        weight,
        output,
        columns,
        kN,
        dilation,
        step,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int skeleton_conv_backward_input(
    at::Tensor input,
    at::Tensor output,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor weight,
    at::Tensor columns,
    int kN,
    float dilation,
    int step,
    int im2col_step) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(weight.type().is_cuda(), "weight tensor is not on GPU!");
    return skeleton_conv_backward_input_cuda(
        input,
        output,
        gradOutput,
        gradInput,
        weight,
        columns,
        kN,
        dilation,
        step,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int skeleton_conv_backward_filter(
    at::Tensor input,
    at::Tensor output,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    int kN,
    float dilation,
    int step,
    float scale,
    int im2col_step) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    return skeleton_conv_backward_parameters_cuda(
        input,
        output,
        gradOutput,
        gradWeight,
        columns,
        kN,
        dilation,
        step,
        scale,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
} // namespace continuous
