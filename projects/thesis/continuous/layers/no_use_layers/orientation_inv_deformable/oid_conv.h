// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace continuous {

#ifdef WITH_CUDA
int oid_conv_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor output,
    at::Tensor columns,
    at::Tensor ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    int im2col_step);

int oid_conv_backward_input_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor gradOffset,
    at::Tensor weight,
    at::Tensor columns,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    int im2col_step);

int oid_conv_backward_parameters_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    at::Tensor ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    float scale,
    int im2col_step);

#endif

inline int oid_conv_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor output,
    at::Tensor columns,
    at::Tensor ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    int im2col_step) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(weight.type().is_cuda(), "weight tensor is not on GPU!");
    TORCH_CHECK(offset.type().is_cuda(), "offset tensor is not on GPU!");
    return oid_conv_forward_cuda(
        input,
        weight,
        offset,
        output,
        columns,
        ones,
        kW,
        kH,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH,
        group,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int oid_conv_backward_input(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor gradOffset,
    at::Tensor weight,
    at::Tensor columns,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    int im2col_step) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(weight.type().is_cuda(), "weight tensor is not on GPU!");
    TORCH_CHECK(offset.type().is_cuda(), "offset tensor is not on GPU!");
    return oid_conv_backward_input_cuda(
        input,
        offset,
        gradOutput,
        gradInput,
        gradOffset,
        weight,
        columns,
        kW,
        kH,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH,
        group,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int oid_conv_backward_filter(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    at::Tensor ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    float scale,
    int im2col_step) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(offset.type().is_cuda(), "offset tensor is not on GPU!");
    return oid_conv_backward_parameters_cuda(
        input,
        offset,
        gradOutput,
        gradWeight,
        columns,
        ones,
        kW,
        kH,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH,
        group,
        scale,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
} // namespace continuous
