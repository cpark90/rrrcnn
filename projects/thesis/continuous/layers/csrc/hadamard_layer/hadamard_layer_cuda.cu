// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_layer_cuda.cpp
// Original license: Apache 2.0

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_layer_cuda.c
// Original license: Apache 2.0

#include <torch/types.h>

#include "hadamard_layer.h"

#include <cmath>
#include <vector>

namespace continuous {

void hadamard_layer_im2col(
    const at::Tensor data_im,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int batch_size,
    at::Tensor data_out);

void hadamard_layer_col2im(
    const at::Tensor data_out,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int batch_size,
    at::Tensor grad_im);

void shape_check_hadamard_layer(
    at::Tensor input,
    at::Tensor* gradOutput,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW) {
  TORCH_CHECK(
      kH > 0 && kW > 0,
      "kernel size should be greater than zero, but got kH: %d, kW: %d",
      kH, kW);

  TORCH_CHECK(
      dW > 0 && dH > 0,
      "stride should be greater than zero, but got dH: %d dW: %d",
      dH,
      dW);

  TORCH_CHECK(
      dilationW > 0 && dilationH > 0,
      "dilation should be greater than zero, but got dilationH: %d dilationW: %d",
      dilationH,
      dilationW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "3D or 4D input tensor expected but got: %s",
      ndim);

  long nInputPlane = input.size(dimf);
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);

  long nOutputPlane = input.size(dimf);

  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (3 - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (3 - 1) + 1)) / dW + 1;


  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane,
        inputHeight,
        inputWidth,
        nOutputPlane,
        outputHeight,
        outputWidth);

  TORCH_CHECK(
      input.size(1) == nInputPlane,
      "invalid number of input planes, expected: %d, but got: %d",
      nInputPlane,
      input.size(1));

  TORCH_CHECK(
      (inputHeight >= 3 && inputWidth >= 3),
      "input image is smaller than kernel");

  if (gradOutput != NULL) {
    dimh++;
    dimw++;
    dimf++;
    TORCH_CHECK(
        gradOutput->size(dimf) == nOutputPlane,
        "invalid number of gradOutput planes, expected: %d, but got: %d",
        nOutputPlane,
        gradOutput->size(dimf));

    TORCH_CHECK(
        (gradOutput->size(dimh) == outputHeight &&
         gradOutput->size(dimw) == outputWidth),
        "invalid size of gradOutput, expected height: %d width: %d , but "
        "got height: %d width: %d",
        outputHeight,
        outputWidth,
        gradOutput->size(dimh),
        gradOutput->size(dimw));
  }
}

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
    int dilationH) {
  shape_check_hadamard_layer(
      input,
      NULL,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW);

  input = input.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.unsqueeze_(0);
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = output.size(2);
  long nKernelPlane = kH * kW;

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (3 - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (3 - 1) + 1)) / dH + 1;


  output = output.view({nKernelPlane,
                        batchSize,
                        nOutputPlane,
                        outputHeight,
                        outputWidth});

  input = input.view({batchSize,
                      nInputPlane,
                      inputHeight,
                      inputWidth});

  at::Tensor output_buffer = at::zeros(
      {nKernelPlane,
       batchSize,
       nOutputPlane,
       outputHeight,
       outputWidth},
      output.options());

  hadamard_layer_im2col(
      input,
      nInputPlane,
      inputHeight,
      inputWidth,
      kH,
      kW,
      padH,
      padW,
      dH,
      dW,
      dilationH,
      dilationW,
      batchSize,
      output_buffer);

  output.copy_(output_buffer);

  if (batch == 0) {
    output = output.view({nKernelPlane, nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

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
    int dilationH) {
  shape_check_hadamard_layer(
      input,
      &gradOutput,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW);

  input = input.contiguous();
  gradOutput = gradOutput.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    gradOutput = gradOutput.view(
        {gradOutput.size(0), 1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradOutput.size(2);
  long nKernelPlane = kW * kH;

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (3 - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (3 - 1) + 1)) / dH + 1;

  // change order of grad output
  gradOutput = gradOutput.view({nKernelPlane,
                                batchSize,
                                nOutputPlane,
                                outputHeight,
                                outputWidth});
  gradInput = gradInput.view({batchSize,
                              nInputPlane,
                              inputHeight,
                              inputWidth});

  hadamard_layer_col2im(
      gradOutput,
      nInputPlane,
      inputHeight,
      inputWidth,
      kH,
      kW,
      padH,
      padW,
      dH,
      dW,
      dilationH,
      dilationW,
      batchSize,
      gradInput);

  if (batch == 0) {
    gradOutput = gradOutput.view({nKernelPlane, nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}
} // namespace continuous
