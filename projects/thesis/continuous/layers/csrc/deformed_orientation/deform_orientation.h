// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace continuous {

#ifdef WITH_CUDA
int deform_orientation_forward_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor output,
    int group,
    int osize,
    int im2col_step);

int deform_orientation_backward_input_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor gradOffset,
    int group,
    int osize,
    int im2col_step,
    int gtype);
#endif

inline int deform_orientation_forward(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor output,
    int group,
    int osize,
    int im2col_step) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(offset.type().is_cuda(), "offset tensor is not on GPU!");
    return deform_orientation_forward_cuda(
        input,
        offset,
        output,
        group,
        osize,
        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

inline int deform_orientation_backward_input(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor gradOffset,
    int group,
    int osize,
    int im2col_step,
    int gtype) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(input.type().is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(offset.type().is_cuda(), "offset tensor is not on GPU!");
    return deform_orientation_backward_input_cuda(
        input,
        offset,
        gradOutput,
        gradInput,
        gradOffset,
        group,
        osize,
        im2col_step,
        gtype);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
} // namespace continuous
