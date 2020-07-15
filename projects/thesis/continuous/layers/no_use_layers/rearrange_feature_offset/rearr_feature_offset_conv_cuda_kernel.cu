// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu
// Original license: Apache 2.0
// clang-format off

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <THC/THCAtomics.cuh>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)


namespace {

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

}

template <typename scalar_t>
__device__ scalar_t deformable_im2col_bilinear(
    const scalar_t* bottom_data,
    const int data_width,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int h,
    const int w,
    const int height,
    const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int height,
    const int width,
    const scalar_t* im_data,
    const int data_width,
    const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void rearr_feature_offset_conv_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_im,
    const scalar_t* data_offset,
    const int height,
    const int width,
    const int grid_h,
    const int grid_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int batch_size,
    const int num_channels,
    const int height_col,
    const int width_col,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const scalar_t step_h = ((float)height - 1) / ((float)grid_h - 1);
    const scalar_t step_w = ((float)width - 1) / ((float)grid_w - 1);

    const int h_in = h_col * stride_h;
    const int w_in = w_col * stride_w;

    scalar_t* data_col_ptr = data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const scalar_t* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t* data_offset_ptr = data_offset +
        b_col * 2 * num_channels * grid_h * grid_w;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((c_im) * grid_h + h_in + i * dilation_h) * grid_w + w_in + j * dilation_w;
        const int data_offset_w_ptr =
            ((num_channels + c_im) * grid_h + h_in + i * dilation_h) * grid_w + w_in + j * dilation_w;

        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];

        scalar_t h_im = (h_in + i * dilation_h + offset_h) * step_h;
        scalar_t w_im = (w_in + j * dilation_w + offset_w) * step_w;
        if (h_im < 0) h_im = 0;
        else if (h_im > height - 1) h_im = height - 1;
        if (w_im < 0) w_im = 0;
        else if (w_im > width - 1) w_im = width - 1;

        /*scalar_t val = static_cast<scalar_t>(0);
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val = deformable_im2col_bilinear(
              data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;*/

        *data_col_ptr = deformable_im2col_bilinear(
              data_im_ptr, width, height, width, h_im, w_im);
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}


template <typename scalar_t>
__global__ void rearr_feature_offset_conv_col2im_gpu_kernel(
    const int n,
    const scalar_t* data_col,
    const scalar_t* data_offset,
    const int channels,
    const int height,
    const int width,
    const int grid_h,
    const int grid_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int batch_size,
    const int height_col,
    const int width_col,
    scalar_t* grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;

    const scalar_t step_h = ((float)height - 1) / ((float)grid_h - 1);
    const scalar_t step_w = ((float)width - 1) / ((float)grid_w - 1);

    const int h_in = h_out * stride_h;
    const int w_in = w_out * stride_w;

    const scalar_t* data_offset_ptr = data_offset +
        b * 2 * channels * grid_h * grid_w;
    const int data_offset_h_ptr =
        ((c) * grid_h + h_in + i * dilation_h) * grid_w + w_in + j * dilation_w;
    const int data_offset_w_ptr =
        ((channels + c) * grid_h + h_in + i * dilation_h) * grid_w + w_in + j * dilation_w;

    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];

    scalar_t cur_inv_h_data = (h_in + i * dilation_h + offset_h) * step_h;
    scalar_t cur_inv_w_data = (w_in + j * dilation_w + offset_w) * step_w;
    if (cur_inv_h_data < 0) cur_inv_h_data = 0;
    else if (cur_inv_h_data > height - 1) cur_inv_h_data = height - 1;
    if (cur_inv_w_data < 0) cur_inv_w_data = 0;
    else if (cur_inv_w_data > width - 1) cur_inv_w_data = width - 1;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight = get_gradient_weight(
              cur_inv_h_data,
              cur_inv_w_data,
              cur_h + dy,
              cur_w + dx,
              height,
              width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}


namespace continuous {

void rearr_feature_offset_conv_im2col(
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int gsize_h,
    const int gsize_w,
    const int ksize_h,
    const int ksize_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    at::Tensor data_col) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col =
      (gsize_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (gsize_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;

  at::cuda::CUDAGuard device_guard(data_im.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.type(), "rearr_feature_offset_conv_im2col_gpu", ([&] {
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* data_col_ = data_col.data_ptr<scalar_t>();

        rearr_feature_offset_conv_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_im_,
            data_offset_,
            height,
            width,
            gsize_h,
            gsize_w,
            ksize_h,
            ksize_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            channels,
            height_col,
            width_col,
            data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in rearr_feature_offset_conv_im2col: %s\n", cudaGetErrorString(err));
  }
}


void rearr_feature_offset_conv_col2im(
    const at::Tensor data_col,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int gsize_h,
    const int gsize_w,
    const int ksize_h,
    const int ksize_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    at::Tensor grad_im) {
  // todo: make sure parallel_imgs is passed in correctly
  int height_col =
      (gsize_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (gsize_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;

  at::cuda::CUDAGuard device_guard(data_col.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.type(), "rearr_feature_offset_conv_col2im_gpu", ([&] {
        const scalar_t* data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* grad_im_ = grad_im.data_ptr<scalar_t>();

        rearr_feature_offset_conv_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_col_,
            data_offset_,
            channels,
            height,
            width,
            gsize_h,
            gsize_w,
            ksize_h,
            ksize_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            height_col,
            width_col,
            grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in rearr_feature_offset_conv_col2im: %s\n", cudaGetErrorString(err));
  }
}

} // namespace continuous

