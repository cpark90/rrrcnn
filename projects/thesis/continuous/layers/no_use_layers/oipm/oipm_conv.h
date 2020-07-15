// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace continuous {

#ifdef WITH_CUDA
int oipm_conv_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor output,
    at::Tensor columns,
    at::Tensor ones,
    int gW,
    int gH,
    int kN,
    float shiftW,
    float shiftH,
    float dilation,
    int group,
    int im2col_step);

int oipm_conv_backward_input_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor weight,
    at::Tensor columns,
    int gW,
    int gH,
    int kN,
    float shiftW,
    float shiftH,
    float dilation,
    int group,
    int im2col_step);

int oipm_conv_backward_parameters_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    at::Tensor ones,
    int gW,
    int gH,
    int kN,
    float shiftW,
    float shiftH,
    float dilation,
    int group,
    float scale,
    int im2col_step);

#endif

inline int oipm_conv_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor output,
    at::Tensor columns,
    at::Tensor ones,
    int gW,
    int gH,
    int kN,
    float shiftW,
    float shiftH,
    float dilation,
    int group,
    int im2col_step) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(offset.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(weight.type().is_cuda(), "weight tensor is not on GPU!");
    return oipm_conv_forward_cuda(
        input,
        weight,
        offset,
        output,
        columns,
        ones,
        gW,
        gH,
        kN,
        shiftW,
        shiftH,
        dilation,
        group,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int oipm_conv_backward_input(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor weight,
    at::Tensor columns,
    int gW,
    int gH,
    int kN,
    float shiftW,
    float shiftH,
    float dilation,
    int group,
    int im2col_step) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(offset.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(weight.type().is_cuda(), "weight tensor is not on GPU!");
    return oipm_conv_backward_input_cuda(
        input,
        offset,
        gradOutput,
        gradInput,
        weight,
        columns,
        gW,
        gH,
        kN,
        shiftW,
        shiftH,
        dilation,
        group,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int oipm_conv_backward_filter(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    at::Tensor ones,
    int gW,
    int gH,
    int kN,
    float shiftW,
    float shiftH,
    float dilation,
    int group,
    float scale,
    int im2col_step) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(offset.type().is_cuda(), "input tensor is not on GPU!");
    return oipm_conv_backward_parameters_cuda(
        input,
        offset,
        gradOutput,
        gradWeight,
        columns,
        ones,
        gW,
        gH,
        kN,
        shiftW,
        shiftH,
        dilation,
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