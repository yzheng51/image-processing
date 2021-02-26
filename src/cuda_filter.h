/**
 * @file cuda_filter.h
 * @yzheng
 * @CUDA mosaic filter
 * @version 0.1
 * @date 2019-05-07
 *
 * @copyright Copyright (c) 2019, yzheng
 *
 */
#ifndef __CUDA_FILTER_H__
#define __CUDA_FILTER_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void checkCUDAError(const char *); // memory check

void mosaic_transform_cuda(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c);

// one thread per block
__global__ void mosaic_filter_cpb(uchar3 *image, uchar3 *image_output, float3 *average, int cols, int rows);

// one thread per block with z axis
__global__ void mosaic_filter_cpb_z(uchar3 *image, float3 *temp_average, int cols, int rows, int c);
__global__ void mosaic_out_cpb_z(float3 *temp_average, uchar3 *image_output, float3 *average, int cols, int rows, int c);

// 32x32 mosaic cells per block
__global__ void mosaic_filter_ccmpb(uchar3 *image, uchar3 *image_output, float3 *average, int cols, int rows, int c);

#endif
