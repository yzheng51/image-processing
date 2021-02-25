/**
 * @file filter.h
 * @yzheng
 * @mosaic filter
 *
 * @version 0.1
 * @date 2019-03-09
 *
 * @copyright Copyright (c) 2019, yzheng
 *
 */
#ifndef __FILTER_H__
#define __FILTER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "libppm.h"

// image average value calculation and mosaic filter and their OpenMP version
void mosaic_transform(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c);
void mosaic_transform_omp(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c);

#ifdef __cplusplus
}
#endif

#endif
