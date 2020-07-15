// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp
// Original license: Apache 2.0

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c
// Original license: Apache 2.0

#include <torch/types.h>

#include "rearr_conv.h"

#include <cmath>
#include <vector>

namespace continuous {

void rearrange_im2col(
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int deformable_group,
    at::Tensor data_col);

void rearrange_col2im(
    const at::Tensor data_col,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int deformable_group,
    at::Tensor grad_im);

void rearrange_col2im_coord(
    const at::Tensor data_col,
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int deformable_group,
    at::Tensor grad_offset);

void shape_check_rearr(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor* gradOutput,
    at::Tensor weight,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int group,
    int deformable_group) {
  TORCH_CHECK(
      weight.ndimension() == 4,
      "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
      "but got: %s",
      weight.ndimension());

  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  TORCH_CHECK(
      kW > 0 && kH > 0,
      "kernel size should be greater than zero, but got kH: %d kW: %d",
      kH,
      kW);

  TORCH_CHECK(
      (weight.size(2) == kH && weight.size(3) == kW),
      "kernel size should be consistent with weight, ",
      "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d",
      kH,
      kW,
      weight.size(2),
      weight.size(3));

  TORCH_CHECK(
      dW > 0 && dH > 0,
      "stride should be greater than zero, but got dH: %d dW: %d",
      dH,
      dW);

  TORCH_CHECK(
      dilationW > 0 && dilationH > 0,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
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

  long nInputPlane = weight.size(1) * group;
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  TORCH_CHECK(
      nInputPlane % deformable_group == 0,
      "input channels must divide deformable group size");

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
      (inputHeight >= kH && inputWidth >= kW),
      "input image is smaller than kernel");

  TORCH_CHECK(
      (offset.size(2) == outputHeight && offset.size(3) == outputWidth),
      "invalid spatial size of offset, expected height: %d width: %d, but "
      "got height: %d width: %d",
      outputHeight,
      outputWidth,
      offset.size(2),
      offset.size(3));

  TORCH_CHECK(
      (offset.size(1) == deformable_group * 2 * kH * kW),
      "invalid number of channels of offset");

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

int rearr_conv_forward_cuda(
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
    int deformable_group,
    int im2col_step) {
  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  shape_check_rearr(
      input,
      offset,
      NULL,
      weight,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      group,
      deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
  weight = weight.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.unsqueeze_(0);
    offset.unsqueeze_(0);
  }

  // todo: assert batchsize dividable by im2col_step

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  output = output.view({batchSize / im2col_step,
                        im2col_step,
                        nOutputPlane,
                        outputHeight,
                        outputWidth});
  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones = at::ones({outputHeight, outputWidth}, input.options());
  }

  input = input.view({batchSize / im2col_step,
                      im2col_step,
                      nInputPlane,
                      inputHeight,
                      inputWidth});
  offset = offset.view({batchSize / im2col_step,
                        im2col_step,
                        deformable_group * 2 * kH * kW,
                        outputHeight,
                        outputWidth});

  at::Tensor output_buffer = at::zeros(
      {batchSize / im2col_step,
       nOutputPlane,
       im2col_step * outputHeight,
       outputWidth},
      output.options());

  output_buffer = output_buffer.view({output_buffer.size(0),
                                      group,
                                      output_buffer.size(1) / group,
                                      output_buffer.size(2),
                                      output_buffer.size(3)});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    rearrange_im2col(
        input[elt],
        offset[elt],
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
        im2col_step,
        deformable_group,
        columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group,
                          weight.size(0) / group,
                          weight.size(1),
                          weight.size(2),
                          weight.size(3)});

    for (int g = 0; g < group; g++) {
      output_buffer[elt][g] = output_buffer[elt][g]
                                  .flatten(1)
                                  .addmm_(weight[g].flatten(1), columns[g])
                                  .view_as(output_buffer[elt][g]);
    }
  }

  output_buffer =
      output_buffer.view({output_buffer.size(0),
                          output_buffer.size(1) * output_buffer.size(2),
                          output_buffer.size(3),
                          output_buffer.size(4)});

  output_buffer = output_buffer.view({batchSize / im2col_step,
                                      nOutputPlane,
                                      im2col_step,
                                      outputHeight,
                                      outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  return 1;
}

int rearr_conv_backward_input_cuda(
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
    int deformable_group,
    int im2col_step) {
  shape_check_rearr(
      input,
      offset,
      &gradOutput,
      weight,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      group,
      deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();
  weight = weight.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
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
  gradOffset = gradOffset.view({batchSize / im2col_step,
                                im2col_step,
                                deformable_group * 2 * kH * kW,
                                outputHeight,
                                outputWidth});
  offset = offset.view({batchSize / im2col_step,
                        im2col_step,
                        deformable_group * 2 * kH * kW,
                        outputHeight,
                        outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    // divide into groups
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group,
                          weight.size(0) / group,
                          weight.size(1),
                          weight.size(2),
                          weight.size(3)});
    gradOutput = gradOutput.view({gradOutput.size(0),
                                  group,
                                  gradOutput.size(1) / group,
                                  gradOutput.size(2),
                                  gradOutput.size(3),
                                  gradOutput.size(4)});

    for (int g = 0; g < group; g++) {
      columns[g] = columns[g].addmm_(
          weight[g].flatten(1).transpose(0, 1),
          gradOutput[elt][g].flatten(1),
          0.0f,
          1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradOutput = gradOutput.view({gradOutput.size(0),
                                  gradOutput.size(1) * gradOutput.size(2),
                                  gradOutput.size(3),
                                  gradOutput.size(4),
                                  gradOutput.size(5)});

    rearrange_col2im_coord(
        columns,
        input[elt],
        offset[elt],
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
        im2col_step,
        deformable_group,
        gradOffset[elt]);

    rearrange_col2im(
        columns,
        offset[elt],
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
        im2col_step,
        deformable_group,
        gradInput[elt]);
  }

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  gradOffset = gradOffset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
    gradOffset =
        gradOffset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  return 1;
}

int rearr_conv_backward_parameters_cuda(
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
    int deformable_group,
    float scale,
    int im2col_step) {
  // todo: transpose and reshape outGrad
  // todo: reshape columns
  // todo: add im2col_step as input

  shape_check_rearr(
      input,
      offset,
      &gradOutput,
      gradWeight,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      group,
      deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
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

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
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
  offset = offset.view({batchSize / im2col_step,
                        im2col_step,
                        deformable_group * 2 * kH * kW,
                        outputHeight,
                        outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    rearrange_im2col(
        input[elt],
        offset[elt],
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
        im2col_step,
        deformable_group,
        columns);

    // divide into group
    gradOutputBuffer = gradOutputBuffer.view({gradOutputBuffer.size(0),
                                              group,
                                              gradOutputBuffer.size(1) / group,
                                              gradOutputBuffer.size(2),
                                              gradOutputBuffer.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    gradWeight = gradWeight.view({group,
                                  gradWeight.size(0) / group,
                                  gradWeight.size(1),
                                  gradWeight.size(2),
                                  gradWeight.size(3)});

    for (int g = 0; g < group; g++) {
      gradWeight[g] = gradWeight[g]
                          .flatten(1)
                          .addmm_(
                              gradOutputBuffer[elt][g].flatten(1),
                              columns[g].transpose(1, 0),
                              1.0,
                              scale)
                          .view_as(gradWeight[g]);
    }
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0),
         gradOutputBuffer.size(1) * gradOutputBuffer.size(2),
         gradOutputBuffer.size(3),
         gradOutputBuffer.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradWeight = gradWeight.view({gradWeight.size(0) * gradWeight.size(1),
                                  gradWeight.size(2),
                                  gradWeight.size(3),
                                  gradWeight.size(4)});
  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}
} // namespace continuous
