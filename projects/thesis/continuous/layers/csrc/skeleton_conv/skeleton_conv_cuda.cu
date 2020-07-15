// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp
// Original license: Apache 2.0

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c
// Original license: Apache 2.0

#include <torch/types.h>

#include "skeleton_conv.h"

#include <cmath>
#include <vector>

namespace continuous {

void skeleton_conv_im2col(
    const at::Tensor data_im,
    const int channels,
    const int height,
    const int width,
    const int ksize,
    const float dilation,
    const int step,
    const int parallel_imgs,
    at::Tensor data_col);

void skeleton_conv_col2im(
    const at::Tensor data_col,
    const int channels,
    const int height,
    const int width,
    const int ksize,
    const float dilation,
    const int step,
    const int parallel_imgs,
    at::Tensor grad_im);

void shape_check_skeleton(
    at::Tensor input,
    at::Tensor output,
    at::Tensor* gradOutput,
    at::Tensor weight,
    int kN,
    float dilation) {
  TORCH_CHECK(
      weight.ndimension() == 3,
      "4D weight tensor (nOutputPlane,nInputPlane,kN) expected, "
      "but got: %s",
      weight.ndimension());

  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  TORCH_CHECK(
      kN > 0,
      "kernel size should be greater than zero, but got kH: %d",
      kN);

  TORCH_CHECK(
      (weight.size(2) == kN),
      "kernel size should be consistent with weight, ",
      "but got kN: %d weight.size(2): %d",
      kN,
      weight.size(2));

  TORCH_CHECK(
      dilation > 0,
      "dilation should be greater than zero, but got dilation: %f",
      dilation);

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

  long nOutputPlane = output.size(1);
  long outputHeight = output.size(dimh);
  long outputWidth = output.size(dimw);


  TORCH_CHECK(
      inputHeight == outputHeight && inputWidth == outputWidth,
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

int skeleton_conv_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    at::Tensor columns,
    int kN,
    float dilation,
    int step,
    int im2col_step) {
  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  shape_check_skeleton(
      input,
      output,
      NULL,
      weight,
      kN,
      dilation);

  input = input.contiguous();
  weight = weight.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.unsqueeze_(0);
  }

  // todo: assert batchsize dividable by im2col_step

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputHeight = output.size(2);
  long outputWidth = output.size(3);

  output = output.view({batchSize / im2col_step,
                        im2col_step,
                        nOutputPlane,
                        outputHeight,
                        outputWidth});
  columns = at::zeros(
      {nInputPlane * kN, im2col_step * outputHeight * outputWidth},
      input.options());

  input = input.view({batchSize / im2col_step,
                      im2col_step,
                      nInputPlane,
                      inputHeight,
                      inputWidth});

  at::Tensor output_buffer = at::zeros(
      {batchSize / im2col_step,
       nOutputPlane,
       im2col_step * outputHeight,
       outputWidth},
      output.options());

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    skeleton_conv_im2col(
        input[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        kN,
        dilation,
        step,
        im2col_step,
        columns);

    output_buffer[elt] = output_buffer[elt]
                                .flatten(1)
                                .addmm_(weight.flatten(1), columns)
                                .view_as(output_buffer[elt]);
  }

  output_buffer = output_buffer.view({batchSize / im2col_step,
                                      nOutputPlane,
                                      im2col_step,
                                      outputHeight,
                                      outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

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
    int im2col_step) {
  shape_check_skeleton(
      input,
      output,
      &gradOutput,
      weight,
      kN,
      dilation);

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
  long outputHeight = output.size(2);
  long outputWidth = output.size(3);

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  columns = at::zeros(
      {nInputPlane * kN, im2col_step * outputHeight * outputWidth},
      input.options());

  // change order of grad output
  gradOutput = gradOutput.view({batchSize / im2col_step,
                                im2col_step,
                                nOutputPlane,
                                outputHeight,
                                outputWidth});
  gradOutput.transpose_(1, 2);

  gradInput = gradInput.view({batchSize / im2col_step,
                              im2col_step,
                              nInputPlane,
                              inputHeight,
                              inputWidth});
  input = input.view({batchSize / im2col_step,
                      im2col_step,
                      nInputPlane,
                      inputHeight,
                      inputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    columns = columns.addmm_(
        weight.flatten(1).transpose(0, 1),
        gradOutput[elt].flatten(1),
        0.0f,
        1.0f);

    skeleton_conv_col2im(
        columns,
        nInputPlane,
        inputHeight,
        inputWidth,
        kN,
        dilation,
        step,
        im2col_step,
        gradInput[elt]);
  }

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

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
    int im2col_step) {
  // todo: transpose and reshape outGrad
  // todo: reshape columns
  // todo: add im2col_step as input

  shape_check_skeleton(
      input,
      output,
      &gradOutput,
      gradWeight,
      kN,
      dilation);

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
  long outputHeight = output.size(2);
  long outputWidth = output.size(3);

  columns = at::zeros(
      {nInputPlane * kN, im2col_step * outputHeight * outputWidth},
      input.options());

  gradOutput = gradOutput.view({batchSize / im2col_step,
                                im2col_step,
                                nOutputPlane,
                                outputHeight,
                                outputWidth});
  gradOutput.transpose_(1, 2);

  at::Tensor gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer = gradOutputBuffer.view({batchSize / im2col_step,
                                            nOutputPlane,
                                            im2col_step,
                                            outputHeight,
                                            outputWidth});
  gradOutputBuffer.copy_(gradOutput);
  // gradOutput is not contiguous, so we do reshape (instead of view) next
  gradOutputBuffer = gradOutputBuffer.reshape({batchSize / im2col_step,
                                               nOutputPlane,
                                               im2col_step * outputHeight,
                                               outputWidth});

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize / im2col_step,
                      im2col_step,
                      nInputPlane,
                      inputHeight,
                      inputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    skeleton_conv_im2col(
        input[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        kN,
        dilation,
        step,
        im2col_step,
        columns);

    // divide into group
    gradOutputBuffer = gradOutputBuffer.view({gradOutputBuffer.size(0),
                                              gradOutputBuffer.size(1),
                                              gradOutputBuffer.size(2),
                                              gradOutputBuffer.size(3)});
    columns = columns.view({columns.size(0), columns.size(1)});
    gradWeight = gradWeight.view({gradWeight.size(0),
                                  gradWeight.size(1),
                                  gradWeight.size(2)});

    gradWeight = gradWeight
                        .flatten(1)
                        .addmm_(
                            gradOutputBuffer[elt].flatten(1),
                            columns.transpose(1, 0),
                            1.0,
                            scale)
                        .view_as(gradWeight);

  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

} // namespace continuous
