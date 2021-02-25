/**
 * @file filter.c
 * @yzheng
 * @mosaic filter
 *
 * @version 0.1
 * @date 2019-03-09
 *
 * @copyright Copyright (c) 2019, yzheng
 *
 */
#include <omp.h>
#include <stdlib.h>

#include "filter.h"

/**
 * @Mosaic filter
 *  divide all pixels into several blocks depend on c
 *  for each block sum up all pixel values and compute the average
 *
 * @param pixels_o
 * @param pixels_i
 * @param cols
 * @param rows
 * @param c
 */
void mosaic_transform(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c) {
    int x, y, i, j;                                     // loop counter
    int m_rows = (rows % c) ? rows / c + 1 : rows / c;  // check whether cols (rows) can be divided by c
    int m_cols = (cols % c) ? cols / c + 1 : cols / c;  // if not, plus one to ensure all pixels has been traversed
    int area = cols * rows;
    unsigned long long average_r, average_g, average_b;
    unsigned long long block_sum_r, block_sum_g, block_sum_b;
    double begin, duration;

    average_r = 0; average_g = 0; average_b = 0;

    begin = omp_get_wtime();

    for (x = 0; x < m_rows; ++x) {
        for (y = 0; y < m_cols; ++y) {
            // partial mosaic for indivisible edges parts, and compute area
            int x_rows = (x == m_rows - 1) ? rows - x * c : c;
            int y_cols = (y == m_cols - 1) ? cols - y * c : c;
            int m_area = x_rows * y_cols;

            block_sum_r = 0, block_sum_g = 0, block_sum_b = 0;

            // sum up pixel values in one block
            for (i = 0; i < x_rows; ++i) {
                for (j = 0; j < y_cols; ++j) {
                    int offset = (x * c + i) * cols + (y * c + j);
                    block_sum_r += pixels_i[offset].r;
                    block_sum_g += pixels_i[offset].g;
                    block_sum_b += pixels_i[offset].b;
                }
            }
            // mosaic calculation and assigin to output pixel array
            for (i = 0; i < x_rows; ++i) {
                for (j = 0; j < y_cols; ++j) {
                    int offset = (x * c + i) * cols + (y * c + j);
                    pixels_o[offset].r = (pixval)(block_sum_r / m_area);
                    pixels_o[offset].g = (pixval)(block_sum_g / m_area);
                    pixels_o[offset].b = (pixval)(block_sum_b / m_area);
                }
            }

            average_r += block_sum_r;
            average_g += block_sum_g;
            average_b += block_sum_b;
        }
    }

    // average value of rgb and store in a pixel
    pixel average_color = {
        (pixval)(average_r / area),
        (pixval)(average_g / area),
        (pixval)(average_b / area)
    };
    duration = omp_get_wtime() - begin;

    printf("CPU mode execution time took %.3lf ms\n", 1000 * duration);
    printf("CPU Average image colour red = %d, green = %d, blue = %d\n", average_color.r, average_color.g, average_color.b);
}

/**
 * @Mosaic filter OpenMP version
 *  divide all pixels into several blocks depend on c
 *  for each block sum up all pixel values and compute the average
 *
 * @param pixels_o
 * @param pixels_i
 * @param cols
 * @param rows
 * @param c
 */
void mosaic_transform_omp(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c) {
    int x, y, i, j;                                     // loop counter
    int m_rows = (rows % c) ? rows / c + 1 : rows / c;  // check whether cols (rows) can be divided by c
    int m_cols = (cols % c) ? cols / c + 1 : cols / c;  // if not, plus one to ensure all pixels has been traversed
    int area = cols * rows;
    unsigned long long average_r, average_g, average_b;
    unsigned long long block_sum_r, block_sum_g, block_sum_b;
    double begin, duration;

    average_r = 0; average_g = 0; average_b = 0;

    begin = omp_get_wtime();

#pragma omp parallel for private(x, y, i, j, block_sum_r, block_sum_g, block_sum_b) reduction(+ : average_r, average_g, average_b)
    for (x = 0; x < m_rows; ++x) {
        for (y = 0; y < m_cols; ++y) {
            // partial mosaic for indivisible edges parts, and compute area
            int x_rows = (x == m_rows - 1) ? rows - x * c : c;
            int y_cols = (y == m_cols - 1) ? cols - y * c : c;
            int m_area = x_rows * y_cols;

            block_sum_r = 0, block_sum_g = 0, block_sum_b = 0;

            // sum up pixel values in one block
            for (i = 0; i < x_rows; ++i) {
                for (j = 0; j < y_cols; ++j) {
                    int offset = (x * c + i) * cols + (y * c + j);
                    block_sum_r += pixels_i[offset].r;
                    block_sum_g += pixels_i[offset].g;
                    block_sum_b += pixels_i[offset].b;
                }
            }
            // mosaic calculation and assigin to output pixel array
            for (i = 0; i < x_rows; ++i) {
                for (j = 0; j < y_cols; ++j) {
                    int offset = (x * c + i) * cols + (y * c + j);
                    pixels_o[offset].r = (pixval)(block_sum_r / m_area);
                    pixels_o[offset].g = (pixval)(block_sum_g / m_area);
                    pixels_o[offset].b = (pixval)(block_sum_b / m_area);
                }
            }

            average_r += block_sum_r;
            average_g += block_sum_g;
            average_b += block_sum_b;
        }
    }

    // average value of rgb will be stor in a pixel
    pixel average_color = {
        (pixval)(average_r / area),
        (pixval)(average_g / area),
        (pixval)(average_b / area)
    };
    duration = omp_get_wtime() - begin;

    printf("CPU mode execution time took %.3lf ms\n", 1000 * duration);
    printf("CPU Average image colour red = %d, green = %d, blue = %d\n", average_color.r, average_color.g, average_color.b);
}
