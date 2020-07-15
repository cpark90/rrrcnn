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
__global__ void hadamard_conv_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_im,
    const int height,
    const int width,
    const int kernel_type,
    const int kernel_size,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int step,
    const int batch_size,
    const int num_channels,
    const int height_col,
    const int width_col,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w = index % width_col;
    const int h = (index / width_col) % height_col;
    const int b = (index / width_col / height_col) % batch_size;
    const int c = (index / width_col / height_col / batch_size) % num_channels;
    const int k = (index / width_col / height_col / batch_size / num_channels) % kernel_size;

    const int h_in = h * stride_h - pad_h;
    const int w_in = w * stride_w - pad_w;

    int i = 1;
    int j = 1;

    if (k > 0) {
      int idx = -1;
      if (kernel_type == 0) {
        idx = ((k - 1) * 2 + step) % 8;
      } else if (kernel_type == 1) {
        idx = ((k - 1) + step) % 8;
      } else {
        idx = ((k - 1) * 4 + step) % 8;
      }

      if (idx == 0) {
        i = 0;
        j = 0;
      } else if (idx == 1) {
        i = 0;
        j = 1;
      } else if (idx == 2) {
        i = 0;
        j = 2;
      } else if (idx == 3) {
        i = 1;
        j = 2;
      } else if (idx == 4) {
        i = 2;
        j = 2;
      } else if (idx == 5) {
        i = 2;
        j = 1;
      } else if (idx == 6) {
        i = 2;
        j = 0;
      } else if (idx == 7) {
        i = 1;
        j = 0;
      }
    }
    const int h_im = h_in + i * dilation_h;
    const int w_im = w_in + j * dilation_w;

    scalar_t val = static_cast<scalar_t>(0);
    if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
      const int data_im_ptr =
          ((b * num_channels + c) * height + h_im) * width + w_im;

      val = data_im[data_im_ptr];
    }
    data_col[index] = val;
  }
}


template <typename scalar_t>
__global__ void hadamard_conv_col2im_gpu_kernel(
    const int n,
    const scalar_t* data_col,
    const int channels,
    const int height,
    const int width,
    const int kernel_type,
    const int kernel_size,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int step,
    const int batch_size,
    const int height_col,
    const int width_col,
    scalar_t* grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w = index % width_col;
    const int h = (index / width_col) % height_col;
    const int b = (index / width_col / height_col) % batch_size;
    const int c = (index / width_col / height_col / batch_size) % channels;
    const int k = (index / width_col / height_col / batch_size / channels) % kernel_size;

    const int h_in = h * stride_h - pad_h;
    const int w_in = w * stride_w - pad_w;

    int i = 1;
    int j = 1;

    if (k > 0) {
      int idx = -1;
      if (kernel_type == 0) {
        idx = ((k - 1) * 2 + step) % 8;
      } else if (kernel_type == 1) {
        idx = ((k - 1) + step) % 8;
      } else {
        idx = ((k - 1) * 4 + step) % 8;
      }

      if (idx == 0) {
        i = 0;
        j = 0;
      } else if (idx == 1) {
        i = 0;
        j = 1;
      } else if (idx == 2) {
        i = 0;
        j = 2;
      } else if (idx == 3) {
        i = 1;
        j = 2;
      } else if (idx == 4) {
        i = 2;
        j = 2;
      } else if (idx == 5) {
        i = 2;
        j = 1;
      } else if (idx == 6) {
        i = 2;
        j = 0;
      } else if (idx == 7) {
        i = 1;
        j = 0;
      }
    }

    const int h_im = h_in + i * dilation_h;
    const int w_im = w_in + j * dilation_w;

    if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
      const int data_im_ptr =
          ((b * channels + c) * height + h_im) * width + w_im;

      atomicAdd(grad_im + data_im_ptr, data_col[index]);
    }
  }
}


namespace continuous {

void hadamard_conv_im2col(
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
    at::Tensor data_col) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col =
      (height + 2 * pad_h - (dilation_h * (3 - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (3 - 1) + 1)) / stride_w + 1;
  int num_kernels =
      ksize * channels * parallel_imgs * height_col * width_col;

  at::cuda::CUDAGuard device_guard(data_im.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.type(), "hadamard_conv_im2col_gpu", ([&] {
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        scalar_t* data_col_ = data_col.data_ptr<scalar_t>();

        hadamard_conv_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_im_,
            height,
            width,
            ktype,
            ksize,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            step,
            parallel_imgs,
            channels,
            height_col,
            width_col,
            data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in hadamard_conv_im2col: %s\n", cudaGetErrorString(err));
  }
}


void hadamard_conv_col2im(
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
    at::Tensor grad_im) {
  // todo: make sure parallel_imgs is passed in correctly
  int height_col =
      (height + 2 * pad_h - (dilation_h * (3 - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (3 - 1) + 1)) / stride_w + 1;
  int num_kernels =
      ksize * channels * parallel_imgs * height_col * width_col;

  at::cuda::CUDAGuard device_guard(data_col.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.type(), "hadamard_conv_col2im_gpu", ([&] {
        const scalar_t* data_col_ = data_col.data_ptr<scalar_t>();
        scalar_t* grad_im_ = grad_im.data_ptr<scalar_t>();

        hadamard_conv_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_col_,
            channels,
            height,
            width,
            ktype,
            ksize,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            step,
            parallel_imgs,
            height_col,
            width_col,
            grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in hadamard_conv_col2im: %s\n", cudaGetErrorString(err));
  }
}
} // namespace continuous
