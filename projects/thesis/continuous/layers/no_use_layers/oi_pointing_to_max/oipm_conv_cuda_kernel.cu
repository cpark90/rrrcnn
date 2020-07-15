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
__global__ void oipm_conv_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_im,
    const scalar_t* data_offset,
    const int height,
    const int width,
    const int grid_h,
    const int grid_w,
    const int kernel_n,
    const float shift_h,
    const float shift_w,
    const float step_h,
    const float step_w,
    const float dilation,
    const float dilation_gap,
    const int batch_size,
    const int num_channels,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    // const int k = (index / grid_w / grid_h / batch_size) % kernel_n;
    const int c = index / grid_w / grid_h / batch_size / 1;

    const int w_col = index % grid_w;
    const int h_col = (index / grid_w) % grid_h;
    const int b = (index / grid_w / grid_h) % batch_size;

    const scalar_t* data_im_ptr =
        data_im + (b * num_channels + c) * height * width;

    const scalar_t* data_offset_ptr = data_offset +
        b * 2 * num_channels * grid_h * grid_w;
    const int data_offset_h_ptr =
        ((c) * grid_h + h_col) * grid_w + w_col;
    const int data_offset_w_ptr =
        ((num_channels + c) * grid_h + h_col) * grid_w + w_col;

    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];

    int best_idx = 0;
    scalar_t best_val = -9999;
    for (int k = 0; k < kernel_n; ++k) {
      const float dilation_k = dilation_gap * k;
      const int receptive_n = 1 + 2 * ceil(dilation_k);
      int count = 1;
      if (receptive_n > 1) count = receptive_n * receptive_n - (receptive_n - 2) * (receptive_n - 2);

      scalar_t val = static_cast<scalar_t>(0);
      for (int r = 0; r < count; ++r){
        const scalar_t h_im = (h_col + offset_h) * step_h + shift_h + dilation_k * cos(2 * M_PI / (float)count * r);
        const scalar_t w_im = (w_col + offset_w) * step_w + shift_w + dilation_k * sin(2 * M_PI / (float)count * r);

        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val += deformable_im2col_bilinear(
              data_im_ptr, width, height, width, h_im, w_im);
        }
      }

      if (best_val < val / count) {
        best_idx = k;
        best_val = val / count;
      }

      /*
      const scalar_t h_im = (h_col + offset_h) * step_h + shift_h + dilation_k * cos(2 * M_PI / (float)count * best_idx);
      const scalar_t w_im = (w_col + offset_w) * step_w + shift_w + dilation_k * sin(2 * M_PI / (float)count * best_idx);

      scalar_t val = static_cast<scalar_t>(0);
      if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
        val = deformable_im2col_bilinear(
            data_im_ptr, width, height, width, h_im, w_im);
      }
      */
    }
    data_col[index] = best_val;
  }
}


template <typename scalar_t>
__global__ void oipm_conv_col2im_gpu_kernel(
    const int n,
    const scalar_t* data_im,
    const scalar_t* data_col,
    const scalar_t* data_offset,
    const int channels,
    const int height,
    const int width,
    const int grid_h,
    const int grid_w,
    const int kernel_n,
    const float shift_h,
    const float shift_w,
    const float step_h,
    const float step_w,
    const float dilation,
    const float dilation_gap,
    const int batch_size,
    scalar_t* grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    // const int k = (index / grid_w / grid_h / batch_size) % kernel_n;
    const int c =
        index / grid_w / grid_h / batch_size / 1;
    // compute the start and end of the output

    int w_out = index % grid_w;
    int h_out = (index / grid_w) % grid_h;
    int b = (index / grid_w / grid_h) % batch_size;

    const scalar_t* data_im_ptr =
        data_im + (b * channels + c) * height * width;

    const scalar_t* data_offset_ptr = data_offset +
        b * 2 * channels * grid_h * grid_w;
    const int data_offset_h_ptr =
        ((c) * grid_h + h_out) * grid_w + w_out;
    const int data_offset_w_ptr =
        ((channels + c) * grid_h + h_out) * grid_w + w_out;

    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];

    int best_idx = 0;
    scalar_t best_val = -9999;

    for (int k = 0; k < kernel_n; ++k) {
      const float dilation_k = dilation_gap * k;
      const int receptive_n = 1 + 2 * ceil(dilation_k);
      int count = 1;
      if (receptive_n > 1) count = receptive_n * receptive_n - (receptive_n - 2) * (receptive_n - 2);

      scalar_t val = static_cast<scalar_t>(0);
      for (int r = 0; r < count; ++r){
        const scalar_t h_im = (h_out + offset_h) * step_h + shift_h + dilation_k * cos(2 * M_PI / (float)count * r);
        const scalar_t w_im = (w_out + offset_w) * step_w + shift_w + dilation_k * sin(2 * M_PI / (float)count * r);

        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val += deformable_im2col_bilinear(
              data_im_ptr, width, height, width, h_im, w_im);
        }
      }
      if (best_val < val) {
        best_idx = k;
        best_val = val;
      }
    }

    const float dilation_k = dilation_gap * best_idx;
    const int receptive_n = 1 + 2 * ceil(dilation_k);
    int count = 1;
    if (receptive_n > 1) count = receptive_n * receptive_n - (receptive_n - 2) * (receptive_n - 2);

    const scalar_t cur_top_grad = data_col[index] / count;
    for (int r = 0; r < count; ++r){
      const scalar_t cur_inv_h_data =
        (h_out + offset_h) * step_h + shift_h + dilation_k * cos(2 * M_PI / (float)count * r);
      const scalar_t cur_inv_w_data =
        (w_out + offset_w) * step_w + shift_w + dilation_k * sin(2 * M_PI / (float)count * r);

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
}


namespace detectron2 {

void oipm_conv_im2col(
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int gsize_h,
    const int gsize_w,
    const int ksize,
    const float shift_h,
    const float shift_w,
    const float dilation,
    const int parallel_imgs,
    at::Tensor data_col) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  float step_h = ((float)height - 1 - 2 * shift_h) / ((float)gsize_h - 1);
  float step_w = ((float)width - 1 - 2 * shift_w) / ((float)gsize_w - 1);
  float dilation_gap = 0;
  if (ksize > 1) dilation_gap = dilation / (ksize - 1);
  int num_kernels =
      channels * 1 * gsize_h * gsize_w * parallel_imgs;

  at::cuda::CUDAGuard device_guard(data_im.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.type(), "oipm_conv_im2col_gpu", ([&] {
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* data_col_ = data_col.data_ptr<scalar_t>();

        oipm_conv_im2col_gpu_kernel<<<
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
            ksize,
            shift_h,
            shift_w,
            step_h,
            step_w,
            dilation,
            dilation_gap,
            parallel_imgs,
            channels,
            data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in oipm_conv_im2col: %s\n", cudaGetErrorString(err));
  }
}


void oipm_conv_col2im(
    const at::Tensor data_im,
    const at::Tensor data_col,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int gsize_h,
    const int gsize_w,
    const int ksize,
    const float shift_h,
    const float shift_w,
    const float dilation,
    const int parallel_imgs,
    at::Tensor grad_im) {
  // todo: make sure parallel_imgs is passed in correctly
  float step_h = ((float)height - 1 - 2 * shift_h) / ((float)gsize_h - 1);
  float step_w = ((float)width - 1 - 2 * shift_w) / ((float)gsize_w - 1);
  float dilation_gap = 0;
  if (ksize > 1) dilation_gap = dilation / (ksize - 1);
  int num_kernels =
      channels * 1 * gsize_h * gsize_w * parallel_imgs;

  at::cuda::CUDAGuard device_guard(data_col.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.type(), "oipm_conv_col2im_gpu", ([&] {
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* grad_im_ = grad_im.data_ptr<scalar_t>();

        oipm_conv_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_im_,
            data_col_,
            data_offset_,
            channels,
            height,
            width,
            gsize_h,
            gsize_w,
            ksize,
            shift_h,
            shift_w,
            step_h,
            step_w,
            dilation,
            dilation_gap,
            parallel_imgs,
            grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in oipm_conv_col2im: %s\n", cudaGetErrorString(err));
  }
}

} // namespace detectron2

