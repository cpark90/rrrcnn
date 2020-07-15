// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp
// Original license: Apache 2.0

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c
// Original license: Apache 2.0

#include <torch/types.h>

#include "deformable_by_grad.h"

#include <cmath>
#include <vector>

namespace continuous {

void deformable_by_grad_im2col(
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int gsize_h,
    const int gsize_w,
    const float shift_h,
    const float shift_w,
    const int parallel_imgs,
    const int group,
    at::Tensor output);

void deformable_by_grad_col2im(
    const at::Tensor grad_out,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int gsize_h,
    const int gsize_w,
    const float shift_h,
    const float shift_w,
    const int parallel_imgs,
    const int group,
    at::Tensor grad_im);

void deformable_by_grad_col2im_coord(
    const at::Tensor grad_out,
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int gsize_h,
    const int gsize_w,
    const float shift_h,
    const float shift_w,
    const int parallel_imgs,
    const int group,
    const int gtype,
    at::Tensor grad_offset);

void shape_check_att(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor* gradOutput,
    int gH,
    int gW,
    float shiftH,
    float shiftW,
    int group) {
  int ndim = input.ndimension();
  int dimc = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimc++;
    dimh++;
    dimw++;
  }

  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "3D or 4D input tensor expected but got: %s",
      ndim);

  long nInputPlane = input.size(dimc);
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);

  long nOutputPlane = nInputPlane;
  long outputHeight = gH;
  long outputWidth = gW;

  long nOffsetPlane = offset.size(1);

  TORCH_CHECK(
      (gH == outputHeight && gW == outputWidth),
      "invalid output size, grid height: %d width: %d, but "
      "got output height: %ld width: %ld",
      gH,
      gW,
      outputHeight,
      outputWidth);

  TORCH_CHECK(
      nInputPlane == nOutputPlane,
      "invalid number of input planes and output planes, Input plane: %d, Output plane: %d",
      nInputPlane,
      nOutputPlane);

  TORCH_CHECK(
      (offset.size(2) == gH && offset.size(3) == gW),
      "invalid spatial size of offset, expected height: %d width: %d, but "
      "got height: %d width: %d",
      gH,
      gW,
      offset.size(2),
      offset.size(3));

  TORCH_CHECK(
      (nOffsetPlane * group == nInputPlane),
      "invalid number of channels of offset",
      "offset plane: %ld, input plane: %ld",
      nOffsetPlane,
      nInputPlane);

  if (gradOutput != NULL) {
    TORCH_CHECK(
        gradOutput->size(dimc) == nOutputPlane,
        "invalid number of gradOutput planes, expected: %d, but got: %d",
        nOutputPlane,
        gradOutput->size(dimc));

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

int deformable_by_grad_forward_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor output,
    int gW,
    int gH,
    float shiftW,
    float shiftH,
    int group,
    int im2col_step) {
  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  shape_check_att(
      input,
      offset,
      NULL,
      gH,
      gW,
      shiftH,
      shiftW,
      group);

  input = input.contiguous();
  offset = offset.contiguous();

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

  long nOutputPlane = output.size(1);
  long outputHeight = output.size(2);
  long outputWidth = output.size(3);

  long nOffsetPlane = offset.size(1);

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  output = output.view({batchSize / im2col_step,
                        im2col_step,
                        nOutputPlane,
                        outputHeight,
                        outputWidth});

  input = input.view({batchSize / im2col_step,
                      im2col_step,
                      nInputPlane,
                      inputHeight,
                      inputWidth});
  offset = offset.view({batchSize / im2col_step,
                        im2col_step,
                        nOffsetPlane,
                        gH,
                        gW});

  at::Tensor output_buffer = at::zeros_like(output);

  output_buffer = output_buffer.view({batchSize / im2col_step,
                                            im2col_step,
                                            nOutputPlane,
                                            outputHeight,
                                            outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_by_grad_im2col(
        input[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        gH,
        gW,
        shiftH,
        shiftW,
        im2col_step,
        group,
        output_buffer[elt]);
  }

  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, nOffsetPlane, gH, gW});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  return 1;
}

int deformable_by_grad_backward_input_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor gradOffset,
    int gW,
    int gH,
    float shiftW,
    float shiftH,
    int group,
    int im2col_step,
    int gtype) {
  shape_check_att(
      input,
      offset,
      &gradOutput,
      gH,
      gW,
      shiftH,
      shiftW,
      group);

  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();

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

  long nOutputPlane = nInputPlane;
  long outputHeight = gH;
  long outputWidth = gW;

  long nOffsetPlane = offset.size(1);

  TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
  gradOutput = gradOutput.view({batchSize / im2col_step,
                                im2col_step,
                                nOutputPlane,
                                outputHeight,
                                outputWidth});

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
                                nOffsetPlane,
                                gH,
                                gW});
  offset = offset.view({batchSize / im2col_step,
                        im2col_step,
                        nOffsetPlane,
                        gH,
                        gW});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_by_grad_col2im(
        gradOutput[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        gH,
        gW,
        shiftH,
        shiftW,
        im2col_step,
        group,
        gradInput[elt]);
    deformable_by_grad_col2im_coord(
        gradOutput[elt],
        input[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        gH,
        gW,
        shiftH,
        shiftW,
        im2col_step,
        group,
        gtype,
        gradOffset[elt]);
  }

  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  gradOffset = gradOffset.view(
      {batchSize, nOffsetPlane, gH, gW});
  offset = offset.view(
      {batchSize, nOffsetPlane, gH, gW});

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

} // namespace continuous
