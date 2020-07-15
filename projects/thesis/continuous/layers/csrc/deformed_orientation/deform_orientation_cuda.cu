// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp
// Original license: Apache 2.0

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c
// Original license: Apache 2.0

#include <torch/types.h>

#include "deform_orientation.h"

#include <cmath>
#include <vector>

namespace continuous {

void deform_orientation_im2col(
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int parallel_imgs,
    const int group,
    const int osize,
    at::Tensor output);

void deform_orientation_col2im(
    const at::Tensor grad_out,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int parallel_imgs,
    const int group,
    const int osize,
    at::Tensor grad_im);

void deform_orientation_col2im_coord(
    const at::Tensor grad_out,
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int parallel_imgs,
    const int group,
    const int osize,
    const int gtype,
    at::Tensor grad_offset);

void shape_check_deo(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor* gradOutput,
    int group,
    int osize) {
  int ndim = input.ndimension();
  int dimc = 0;
  int dimh = 1;
  int dimw = 2;
  int dimo = 3;

  if (ndim == 5) {
    dimc++;
    dimh++;
    dimw++;
    dimo++;
  }

  TORCH_CHECK(
      ndim == 4 || ndim == 5,
      "4D or 5D input tensor expected but got: %s",
      ndim);

  long nInputPlane = input.size(dimc);
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long inputOrientation = input.size(dimo);

  long nOutputPlane = nInputPlane;
  long outputHeight = inputHeight;
  long outputWidth = inputWidth;

  long nOffsetPlane = offset.size(1);

  TORCH_CHECK(
      (offset.size(2) == outputHeight && offset.size(3) == outputWidth),
      "invalid spatial size of offset, expected height: %ld width: %ld, but "
      "got height: %d width: %d",
      outputHeight,
      outputWidth,
      offset.size(2),
      offset.size(3));

  TORCH_CHECK(
      (inputOrientation == osize),
      "invalid number of orientation",
      "input orientation: %ld, orientation: %d",
      inputOrientation,
      osize);

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

int deform_orientation_forward_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor output,
    int group,
    int osize,
    int im2col_step) {
  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  shape_check_deo(
      input,
      offset,
      NULL,
      group,
      osize);

  input = input.contiguous();
  offset = offset.contiguous();

  int batch = 1;
  if (input.ndimension() == 4) {
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
  int orientation = input.size(4);

  long nOutputPlane = output.size(1);
  long outputHeight = output.size(2);
  long outputWidth = output.size(3);

  long nOffsetPlane = offset.size(1);
  long offsetHeight = offset.size(2);
  long offsetWidth = offset.size(3);

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");
  TORCH_CHECK((outputHeight == inputHeight && outputWidth == inputWidth), "invalid height width size");

  output = output.view({batchSize / im2col_step,
                        im2col_step,
                        nOutputPlane,
                        outputHeight,
                        outputWidth,
                        orientation});

  input = input.view({batchSize / im2col_step,
                      im2col_step,
                      nInputPlane,
                      inputHeight,
                      inputWidth,
                      orientation});
  offset = offset.view({batchSize / im2col_step,
                        im2col_step,
                        nOffsetPlane,
                        offsetHeight,
                        offsetWidth,
                        orientation});

  at::Tensor output_buffer = at::zeros_like(output);

  output_buffer = output_buffer.view({batchSize / im2col_step,
                                            im2col_step,
                                            nOutputPlane,
                                            outputHeight,
                                            outputWidth,
                                            orientation});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deform_orientation_im2col(
        input[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        im2col_step,
        group,
        osize,
        output_buffer[elt]);
  }

  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth, orientation});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth, orientation});
  offset = offset.view(
      {batchSize, nOffsetPlane, offsetHeight, offsetWidth, orientation});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth, orientation});
    input = input.view({nInputPlane, inputHeight, inputWidth, orientation});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
  }

  return 1;
}

int deform_orientation_backward_input_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor gradOffset,
    int group,
    int osize,
    int im2col_step,
    int gtype) {
  shape_check_deo(
      input,
      offset,
      &gradOutput,
      group,
      osize);

  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2), input.size(3)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2), offset.size(3)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3)});
  }

  long batchSize = input.size(0);

  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);
  int orientation = input.size(4);

  long nOutputPlane = gradOutput.size(1);
  long outputHeight = gradOutput.size(2);
  long outputWidth = gradOutput.size(3);

  long nOffsetPlane = offset.size(1);
  long offsetHeight = offset.size(2);
  long offsetWidth = offset.size(3);

  TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
  gradOutput = gradOutput.view({batchSize / im2col_step,
                                im2col_step,
                                nOutputPlane,
                                outputHeight,
                                outputWidth,
                                orientation});

  gradInput = gradInput.view({batchSize / im2col_step,
                              im2col_step,
                              nInputPlane,
                              inputHeight,
                              inputWidth,
                              orientation});
  input = input.view({batchSize / im2col_step,
                      im2col_step,
                      nInputPlane,
                      inputHeight,
                      inputWidth,
                      orientation});
  gradOffset = gradOffset.view({batchSize / im2col_step,
                                im2col_step,
                                nOffsetPlane,
                                offsetHeight,
                                offsetWidth,
                                orientation});
  offset = offset.view({batchSize / im2col_step,
                        im2col_step,
                        nOffsetPlane,
                        offsetHeight,
                        offsetWidth,
                        orientation});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deform_orientation_col2im_coord(
        gradOutput[elt],
        input[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        im2col_step,
        group,
        osize,
        gtype,
        gradOffset[elt]);

    deform_orientation_col2im(
        gradOutput[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        im2col_step,
        group,
        osize,
        gradInput[elt]);
  }

  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth, orientation});

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth, orientation});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth, orientation});
  gradOffset = gradOffset.view(
      {batchSize, nOffsetPlane, offsetHeight, offsetWidth, orientation});
  offset = offset.view(
      {batchSize, nOffsetPlane, offsetHeight, offsetWidth, orientation});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth, orientation});
    input = input.view({nInputPlane, inputHeight, inputWidth, orientation});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth, orientation});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
    gradOffset =
        gradOffset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
  }

  return 1;
}

} // namespace continuous
