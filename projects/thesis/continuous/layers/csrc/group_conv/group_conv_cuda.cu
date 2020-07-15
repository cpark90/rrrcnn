// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp
// Original license: Apache 2.0

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c
// Original license: Apache 2.0

#include <torch/types.h>

#include "group_conv.h"

#include <cmath>
#include <vector>

namespace continuous {

void group_conv_im2col(
    const at::Tensor data_im,
    const int channels,
    const int height,
    const int width,
    const int ktype,
    const int ksize,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int step,
    const int parallel_imgs,
    at::Tensor data_col);

void group_conv_col2im(
    const at::Tensor data_col,
    const int channels,
    const int height,
    const int width,
    const int ktype,
    const int ksize,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int step,
    const int parallel_imgs,
    at::Tensor grad_im);

void shape_check_group(
    at::Tensor input,
    at::Tensor* gradOutput,
    at::Tensor weight,
    int nK,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW) {
  TORCH_CHECK(
      weight.ndimension() == 3,
      "3D weight tensor (k, nOutputPlane,nInputPlane) expected, "
      "but got: %s",
      weight.ndimension());

  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  TORCH_CHECK(
      nK == 5 || nK == 9 || nK == 3,
      "kernel size should be greater than zero, but got nK: %d",
      nK);

  TORCH_CHECK(
      (weight.size(2) == nK),
      "kernel size should be consistent with weight, ",
      "but got nK: %d weight.size(2): %d",
      nK,
      weight.size(2));

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

  long nInputPlane = weight.size(1);
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);

  long nOutputPlane = weight.size(0);
  long nKernelPlane = weight.size(2);

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
    int step) {
  shape_check_group(
      input,
      NULL,
      weight,
      nK,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW);

  input = input.contiguous();
  weight = weight.contiguous();

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

  long nOutputPlane = weight.size(0);
  long nKernelPlane = weight.size(2);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (3 - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (3 - 1) + 1)) / dH + 1;


  output = output.view({batchSize,
                        nOutputPlane,
                        outputHeight,
                        outputWidth});
  columns = at::zeros(
      {nInputPlane * nKernelPlane, batchSize * outputHeight * outputWidth},
      input.options());

  input = input.view({batchSize,
                      nInputPlane,
                      inputHeight,
                      inputWidth});

  at::Tensor output_buffer = at::zeros(
      {nOutputPlane,
       batchSize,
       outputHeight,
       outputWidth},
      output.options());

  group_conv_im2col(
      input,
      nInputPlane,
      inputHeight,
      inputWidth,
      tK,
      nK,
      padH,
      padW,
      dH,
      dW,
      dilationH,
      dilationW,
      step,
      batchSize,
      columns);

  output_buffer = output_buffer.flatten(1)
                              .addmm_(weight.flatten(1), columns)
                              .view_as(output_buffer);

  output_buffer.transpose_(0, 1);
  output.copy_(output_buffer);

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

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
    int step) {
  shape_check_group(
      input,
      &gradOutput,
      weight,
      nK,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW);

  input = input.contiguous();
  gradOutput = gradOutput.contiguous();
  weight = weight.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);
  long nKernelPlane = weight.size(2);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (3 - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (3 - 1) + 1)) / dH + 1;

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  columns = at::zeros(
      {nInputPlane * nKernelPlane, batchSize * outputHeight * outputWidth},
      input.options());

  // change order of grad output
  gradOutput = gradOutput.view({batchSize,
                                nOutputPlane,
                                outputHeight,
                                outputWidth});
  gradOutput.transpose_(0, 1);

  gradInput = gradInput.view({batchSize,
                              nInputPlane,
                              inputHeight,
                              inputWidth});
  input = input.view({batchSize,
                      nInputPlane,
                      inputHeight,
                      inputWidth});

  columns = columns.addmm_(
      weight.flatten(1).transpose(0, 1),
      gradOutput.flatten(1),
      0.0f,
      1.0f);

  gradOutput.transpose_(0, 1);

  group_conv_col2im(
      columns,
      nInputPlane,
      inputHeight,
      inputWidth,
      tK,
      nK,
      padH,
      padW,
      dH,
      dW,
      dilationH,
      dilationW,
      step,
      batchSize,
      gradInput);

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

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
    float scale) {
  // todo: transpose and reshape outGrad
  // todo: reshape columns
  // todo: add im2col_step as input

  shape_check_group(
      input,
      &gradOutput,
      gradWeight,
      nK,
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
    input = input.view(
        at::IntList({1, input.size(0), input.size(1), input.size(2)}));
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);
  long nKernelPlane = gradWeight.size(2);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (3 - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (3 - 1) + 1)) / dH + 1;

  columns = at::zeros(
      {nInputPlane * nKernelPlane, batchSize * outputHeight * outputWidth},
      input.options());

  gradOutput = gradOutput.view({batchSize,
                                nOutputPlane,
                                outputHeight,
                                outputWidth});

  input = input.view({batchSize,
                      nInputPlane,
                      inputHeight,
                      inputWidth});

  group_conv_im2col(
      input,
      nInputPlane,
      inputHeight,
      inputWidth,
      tK,
      nK,
      padH,
      padW,
      dH,
      dW,
      dilationH,
      dilationW,
      step,
      batchSize,
      columns);

  gradOutput.transpose_(0, 1);

  gradWeight = gradWeight.flatten(1)
                      .addmm_(
                          gradOutput.flatten(1),
                          columns.transpose(0, 1),
                          1.0,
                          scale)
                      .view_as(gradWeight);

  gradOutput.transpose_(0, 1);

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}
} // namespace continuous
