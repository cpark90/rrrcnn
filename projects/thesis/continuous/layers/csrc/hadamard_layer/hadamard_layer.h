// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace continuous {

#ifdef WITH_CUDA
int hadamard_layer_forward_cuda(
    at::Tensor input,
    at::Tensor output,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);

int hadamard_layer_backward_input_cuda(
    at::Tensor input,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);
#endif

inline int hadamard_layer_forward(
    at::Tensor input,
    at::Tensor output,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return hadamard_layer_forward_cuda(
        input,
        output,
        kW,
        kH,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int hadamard_layer_backward_input(
    at::Tensor input,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    return hadamard_layer_backward_input_cuda(
        input,
        gradOutput,
        gradInput,
        kW,
        kH,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
} // namespace continuous
