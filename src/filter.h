/**
 * @file filter.h
 * @yzheng
 * @mosaic filter
 *  only mosaic_transform() and image_average_value() (and its OpenMP version) are used in mosaic.c
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

// data type for color and rgb structure
// lpixval means long pixel value, which can store large integer
// used to sum up pixels and calculate average
// uint32 range from 0-4294967295, which is enough for image with 4096x4096
// so it is waste of memory for using unsigned long long
typedef unsigned long long lpixval;
typedef struct _lpixel {
    lpixval r, g, b;
} lpixel;

// memory allocation and free for lpixels
// init_lpixel() not only allocate memory but also initialise the value to 0
// by calloc() which is used to calculate the rgb value for average
lpixel *init_lpixel(int cols, int rows);
void free_lpixel(lpixel *pixels);

// image average value calculation and mosaic filter and their OpenMP version
void mosaic_transform(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c);
void mosaic_transform_omp(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c);

#ifdef __cplusplus
}
#endif

#endif