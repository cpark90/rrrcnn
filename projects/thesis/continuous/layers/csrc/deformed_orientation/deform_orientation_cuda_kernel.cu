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
    scalar_t o,
    const int osize) {
  int o_low = floor(o);
  int o_high = o_low + 1;

  scalar_t lo = o - o_low;
  scalar_t ho = 1 - lo;

  scalar_t lv = bottom_data[(o_low % osize + osize) % osize];
  scalar_t hv = bottom_data[(o_high % osize + osize) % osize];

  scalar_t val = (ho * lv + lo * hv);
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(
    scalar_t argmax_o,
    const int o) {
  int argmax_o_low = floor(argmax_o);
  int argmax_o_high = argmax_o_low + 1;

  scalar_t weight = 0;
  if (o == argmax_o_low)
    weight = (o + 1 - argmax_o);
  if (o == argmax_o_high)
    weight = (argmax_o + 1 - o);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(
    scalar_t argmax_o,
    const scalar_t* im_data,
    const int osize,
    const int gtype) {

  int argmax_o_low = floor(argmax_o);
  int argmax_o_high = argmax_o_low + 1;

  scalar_t weight = 0;
  scalar_t val = 0;

  weight += -1 * (argmax_o_low + 1 - argmax_o) *
      im_data[(argmax_o_low % osize + osize) % osize];
  val += (argmax_o_low + 1 - argmax_o) *
      im_data[(argmax_o_low % osize + osize) % osize];

  weight += (argmax_o - argmax_o_low) *
      im_data[(argmax_o_high % osize + osize) % osize];
  val += (argmax_o - argmax_o_low) *
      im_data[(argmax_o_high % osize + osize) % osize];

  if (gtype == 1) return weight * tanh(val);
  else if (gtype == 2) return weight * exp(-1 / 2 * val * val);
  else return weight;
}

template <typename scalar_t>
__global__ void deform_orientation_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_im,
    const scalar_t* data_offset,
    const int height,
    const int width,
    const int batch_size,
    const int group,
    const int osize,
    const int num_channels,
    scalar_t* data_out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int offset_channels = num_channels / group;

    const int w_offset = index % width;
    const int h_offset = (index / width) % height;
    const int c_offset = (index / width / height) % offset_channels;
    const int b_offset = (index / width / height / offset_channels) % batch_size;
    const int o_offset = (index / width / height / offset_channels / batch_size);

    scalar_t* data_out_ptr =
        data_out + (((b_offset * num_channels + c_offset * group) * height + h_offset) * width + w_offset) * osize + o_offset;

    for (int idx = 0; idx < group; ++idx) {
      const scalar_t* data_im_ptr =
          data_im + (((b_offset * num_channels + c_offset * group + idx) * height + h_offset) * width + w_offset) * osize;

      const scalar_t* data_offset_ptr = data_offset +
          b_offset * offset_channels * height * width * osize;
      const int data_offset_o_ptr =
          (((c_offset) * height + h_offset) * width + w_offset) * osize + o_offset;
      const scalar_t offset_o = data_offset_ptr[data_offset_o_ptr];

      *data_out_ptr = deformable_im2col_bilinear(
          data_im_ptr, o_offset + offset_o, osize);
      data_out_ptr += height * width * osize;
    }
  }
}


template <typename scalar_t>
__global__ void deform_orientation_col2im_gpu_kernel(
    const int n,
    const scalar_t* grad_out,
    const scalar_t* data_offset,
    const int channels,
    const int height,
    const int width,
    const int batch_size,
    const int group,
    const int osize,
    scalar_t* grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int offset_channels = channels / group;

    const int w_out = index % width;
    const int h_out = (index / width) % height;
    const int g = (index / width / height) % group;
    const int c = (index / width / height / group) % offset_channels;
    const int b = (index / width / height / group / offset_channels) % batch_size;
    const int o = (index / width / height / group / offset_channels / batch_size);

    const int grad_out_ptr =
      (((b * channels + c * group + g) * height + h_out) * width + w_out) * osize + o;

    const scalar_t* data_offset_ptr = data_offset +
        b * offset_channels * height * width * osize;
    const int data_offset_o_ptr =
        (((c) * height + h_out) * width + w_out) * osize + o;
    const scalar_t offset_o = data_offset_ptr[data_offset_o_ptr];

    const scalar_t cur_top_grad = grad_out[grad_out_ptr];

    const int cur_o = (int)(o + offset_o);
    for (int dor = -2; dor <= 2; dor++) {
      if (abs(offset_o - (cur_o + dor)) < 1) {
        int cur_bottom_grad_pos =
            (((b * channels + c * group + g) * height + h_out) * width + w_out) * osize + ((cur_o + dor) % osize + osize) % osize;
        scalar_t weight = get_gradient_weight(o + offset_o, cur_o + dor);
        atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
      }
    }
  }
}


template <typename scalar_t>
__global__ void deform_orientation_col2im_coord_gpu_kernel(
    const int n,
    const scalar_t* grad_out,
    const scalar_t* data_im,
    const scalar_t* data_offset,
    const int channels,
    const int height,
    const int width,
    const int batch_size,
    const int group,
    const int osize,
    const int gtype,
    scalar_t* grad_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    const int offset_channels = channels / group;

    const int w_out = index % width;
    const int h_out = (index / width) % height;
    const int g = (index / width / height) % group;
    const int c = (index / width / height / group) % offset_channels;
    const int b = (index / width / height / group / offset_channels) % batch_size;
    const int o = (index / width / height / group / offset_channels / batch_size);

    const int grad_out_ptr =
      (((b * channels + c * group + g) * height + h_out) * width + w_out) * osize + o;

    const scalar_t* data_im_ptr = data_im +
      (((b * channels + c * group + g) * height + h_out) * width + w_out) * osize;
    const scalar_t* data_offset_ptr = data_offset +
      b * offset_channels * height * width * osize;
    const int data_offset_o_ptr =
      (((c) * height + h_out) * width + w_out) * osize + o;

    const scalar_t offset_o = data_offset_ptr[data_offset_o_ptr];

    const scalar_t weight_o = get_coordinate_weight(
      o + offset_o,
      data_im_ptr,
      osize,
      gtype);
    grad_offset[data_offset_o_ptr] += weight_o * grad_out[grad_out_ptr];
  }
}


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
    at::Tensor data_out) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int num_kernels = osize * channels / group * height * width * parallel_imgs;

  at::cuda::CUDAGuard device_guard(data_im.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.type(), "deform_orientation_im2col_gpu", ([&] {
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* data_out_ = data_out.data_ptr<scalar_t>();

        deform_orientation_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_im_,
            data_offset_,
            height,
            width,
            parallel_imgs,
            group,
            osize,
            channels,
            data_out_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deform_orientation_im2col: %s\n", cudaGetErrorString(err));
  }
}


void deform_orientation_col2im(
    const at::Tensor grad_out,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int parallel_imgs,
    const int group,
    const int osize,
    at::Tensor grad_im) {
  // todo: make sure parallel_imgs is passed in correctly
  int num_kernels =
      osize * channels * height * width * parallel_imgs;

  at::cuda::CUDAGuard device_guard(grad_out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.type(), "deform_orientation_col2im_gpu", ([&] {
        const scalar_t* grad_out_ = grad_out.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* grad_im_ = grad_im.data_ptr<scalar_t>();

        deform_orientation_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            grad_out_,
            data_offset_,
            channels,
            height,
            width,
            parallel_imgs,
            group,
            osize,
            grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deform_orientation_col2im: %s\n", cudaGetErrorString(err));
  }
}


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
    at::Tensor grad_offset) {
  int num_kernels =
      osize * channels * height * width * parallel_imgs;

  at::cuda::CUDAGuard device_guard(grad_out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.type(), "deform_orientation_col2im_coord_gpu", ([&] {
        const scalar_t* grad_out_ = grad_out.data_ptr<scalar_t>();
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* grad_offset_ = grad_offset.data_ptr<scalar_t>();

        deform_orientation_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            grad_out_,
            data_im_,
            data_offset_,
            channels,
            height,
            width,
            parallel_imgs,
            group,
            osize,
            gtype,
            grad_offset_);
      }));
}

} // namespace continuous

