// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace continuous {

#ifdef WITH_CUDA
int group_conv_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    at::Tensor columns,
    int tK,
    int nK,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int step);

int group_conv_backward_input_cuda(
    at::Tensor input,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor weight,
    at::Tensor columns,
    int tK,
    int nK,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int step);

int group_conv_backward_parameters_cuda(
    at::Tensor input,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    int tK,
    int nK,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int step,
    float scale);

#endif

inline int group_conv_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    at::Tensor columns,
    int tK,
    int nK,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int step) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(weight.type().is_cuda(), "weight tensor is not on GPU!");
    return group_conv_forward_cuda(
        input,
        weight,
        output,
        columns,
        tK,
        nK,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH,
        step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int group_conv_backward_input(
    at::Tensor input,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor weight,
    at::Tensor columns,
    int tK,
    int nK,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int step) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(weight.type().is_cuda(), "weight tensor is not on GPU!");
    return group_conv_backward_input_cuda(
        input,
        gradOutput,
        gradInput,
        weight,
        columns,
        tK,
        nK,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH,
        step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int group_conv_backward_filter(
    at::Tensor input,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    int tK,
    int nK,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int step,
    float scale) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    return group_conv_backward_parameters_cuda(
        input,
        gradOutput,
        gradWeight,
        columns,
        tK,
        nK,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH,
        step,
        scale);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
} // namespace continuous
