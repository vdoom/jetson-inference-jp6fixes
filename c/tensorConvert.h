/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __CUDA_TENSOR_PREPROCESSING_H__
#define __CUDA_TENSOR_PREPROCESSING_H__


#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>


/*
 * Downsample and apply mean pixel subtraction, NCHW format
 */
cudaError_t cudaTensorMeanRGB( void* input, imageFormat format, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream );
cudaError_t cudaTensorMeanBGR( void* input, imageFormat format, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream );

/*
 * Downsample and apply pixel normalization, NCHW format
 */
cudaError_t cudaTensorNormRGB( void* input, imageFormat format, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, cudaStream_t stream );
cudaError_t cudaTensorNormBGR( void* input, imageFormat format, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, cudaStream_t stream );

/*
 * Downsample and apply pixel normalization, mean pixel subtraction and standard deviation, NCHW format
 */
cudaError_t cudaTensorNormMeanRGB( void* input, imageFormat format, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream, size_t channelStride=0 );
cudaError_t cudaTensorNormMeanBGR( void* input, imageFormat format, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream, size_t channelStride=0 );

/*
 * Aspect-preserving (letterbox) downsample + pixel normalization, NCHW format.
 * Matches DeepStream maintain-aspect-ratio=1 + symmetric-padding=1: the source is resized by
 * r = min(outW/inW, outH/inH) (bilinear) into a centered region of the output, the remainder
 * filled with padValue (in source pixel units, e.g. 114), then normalized to `range`.  The
 * applied resize ratio and symmetric pad offsets are returned via outScale/outPadX/outPadY so
 * detections can be inverse-mapped back to original-image coordinates.
 */
cudaError_t cudaTensorNormLetterboxRGB( void* input, imageFormat format, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, float padValue, float* outScale, int* outPadX, int* outPadY, cudaStream_t stream );


#endif

