/**
 * @file mf.c
 * @yzheng
 * @mosaic filter
 *  only mosaic_transform() and image_average_value() (and its OPENMP version) are used in mosaic.c
 * @version 0.1
 * @date 2019-03-09
 *
 * @copyright Copyright (c) 2019, yzheng
 *
 */
#include "filter.h"
#include <omp.h>
#include <stdlib.h>

/**
 * @Allocate memory and initialise the rgb value for lpixel matrices
 *
 * @param cols
 * @param rows
 * @return lpixel**
 */
lpixel *init_lpixel(int cols, int rows) {
    lpixel *pixels = NULL;

    pixels = (lpixel *)calloc(rows * cols, sizeof(lpixel));
    // check whether the allocation is successful, the same in loop
    if (pixels == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(1);
    }

    return pixels;
}

/**
 * @Free lpixel matrices
 *
 * @param pixels
 * @param rows
 */
void free_lpixel(lpixel *pixels) {
    free(pixels);
}

/**
 * @Mosaic filter
 *  this algorithm is to create a pixel matrix to store the average rgb
 *  value of the original image. its dimension is (m_rows x is m_cols)
 *  due to average, some data type casting is used here as well as the integer
 *  over flow check
 *
 * @param pixels_o
 * @param pixels_i
 * @param cols
 * @param rows
 * @param c
 */
void mosaic_transform(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c) {
    int i, j;                                           // loop counter
    int mod_cols = cols % c;                            // remain part of width
    int mod_rows = rows % c;                            // remain part of height
    int m_cols = (mod_cols) ? cols / c + 1 : cols / c;  // check whether cols (rows) can be divided by c
    int m_rows = (mod_rows) ? rows / c + 1 : rows / c;  // if not, plus one to ensure enough memory has benn allocated
    lpixval m_area = c * c;
    lpixval area = cols * rows;
    lpixval average_r, average_g, average_b;
    lpixel *lpixels;  // pixel for calculation
    double begin, duration;

    average_r = 0; average_g = 0; average_b = 0;

    // memory allocation and value initialisation
    lpixels = init_lpixel(m_cols, m_rows);

    begin = omp_get_wtime();
    // sum up all rgb values and stor in the average pixel
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            int offset = i * cols + j;
            int avg_offset = i / c * m_cols + j / c;
            lpixels[avg_offset].r += pixels_i[offset].r;
            lpixels[avg_offset].g += pixels_i[offset].g;
            lpixels[avg_offset].b += pixels_i[offset].b;
        }
    }

    /**
     * @image is divided into four parts
     *  |---------------------|---|
     *  |                     |   |
     *  |                     |   |
     *  |                     |   |
     *  |                     |   |
     *  |                     |   |
     *  |                     |   |
     *  |---------------------|---|
     *  |---------------------|---|
     *  top left part: c can divide rows and cols in this part
     *  top right part: c cannot divide rows in this part
     *  bottom left part: c cannot divide cols in this part
     *  bottom right part: c cannot divide rows and cols in this part
     *
     *  it is inefficient to use if clause and calculate inside the loop, also
     *  traverse by row is needed for the bottom part but not all images
     *  requir this kind of calculation which is slow. therefore, the code
     *  in this section is long. However, it is more efficient because the image
     *  can only meet one condition during the process
     */
    if (!mod_cols && !mod_rows) {
        for (i = 0; i < m_rows; ++i) {
            for (j = 0; j < m_cols; ++j) {
                int offset = i * m_cols + j;
                average_r += lpixels[offset].r;
                average_g += lpixels[offset].g;
                average_b += lpixels[offset].b;
                lpixels[offset].r /= m_area;
                lpixels[offset].g /= m_area;
                lpixels[offset].b /= m_area;
            }
        }
    } else if (mod_cols && !mod_rows) {
        for (i = 0; i < m_rows; ++i) {
            for (j = 0; j < m_cols - 1; ++j) {
                int offset = i * m_cols + j;
                average_r += lpixels[offset].r;
                average_g += lpixels[offset].g;
                average_b += lpixels[offset].b;
                lpixels[offset].r /= m_area;
                lpixels[offset].g /= m_area;
                lpixels[offset].b /= m_area;
            }
        }

        j = m_cols - 1;
        for (i = 0; i < m_rows; ++i) {
            int offset = i * m_cols + j;
            average_r += lpixels[offset].r;
            average_g += lpixels[offset].g;
            average_b += lpixels[offset].b;
            lpixels[offset].r /= c * mod_cols;
            lpixels[offset].g /= c * mod_cols;
            lpixels[offset].b /= c * mod_cols;
        }
    } else if (!mod_cols && mod_rows) {
        for (i = 0; i < m_rows - 1; ++i) {
            for (j = 0; j < m_cols; ++j) {
                int offset = i * m_cols + j;
                average_r += lpixels[offset].r;
                average_g += lpixels[offset].g;
                average_b += lpixels[offset].b;
                lpixels[offset].r /= m_area;
                lpixels[offset].g /= m_area;
                lpixels[offset].b /= m_area;
            }
        }

        i = m_rows - 1;
        for (j = 0; j < m_cols; ++j) {
            int offset = i * m_cols + j;
            average_r += lpixels[offset].r;
            average_g += lpixels[offset].g;
            average_b += lpixels[offset].b;
            lpixels[offset].r /= mod_rows * c;
            lpixels[offset].g /= mod_rows * c;
            lpixels[offset].b /= mod_rows * c;
        }
    } else {
        for (i = 0; i < m_rows - 1; ++i) {
            for (j = 0; j < m_cols - 1; ++j) {
                int offset = i * m_cols + j;
                average_r += lpixels[offset].r;
                average_g += lpixels[offset].g;
                average_b += lpixels[offset].b;
                lpixels[offset].r /= m_area;
                lpixels[offset].g /= m_area;
                lpixels[offset].b /= m_area;
            }
        }

        i = m_rows - 1;
        for (j = 0; j < m_cols - 1; ++j) {
            int offset = i * m_cols + j;
            average_r += lpixels[offset].r;
            average_g += lpixels[offset].g;
            average_b += lpixels[offset].b;
            lpixels[offset].r /= mod_rows * c;
            lpixels[offset].g /= mod_rows * c;
            lpixels[offset].b /= mod_rows * c;
        }

        j = m_cols - 1;
        for (i = 0; i < m_rows - 1; ++i) {
            int offset = i * m_cols + j;
            average_r += lpixels[offset].r;
            average_g += lpixels[offset].g;
            average_b += lpixels[offset].b;
            lpixels[offset].r /= c * mod_cols;
            lpixels[offset].g /= c * mod_cols;
            lpixels[offset].b /= c * mod_cols;
        }
        int offset = (m_cols - 1) * m_rows + m_rows - 1;
        average_r += lpixels[offset].r;
        average_g += lpixels[offset].g;
        average_b += lpixels[offset].b;
        lpixels[offset].r /= mod_rows * mod_cols;
        lpixels[offset].g /= mod_rows * mod_cols;
        lpixels[offset].b /= mod_rows * mod_cols;
    }

    // similar to the calculation part, assignment the rgb value back
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            int offset = i * cols + j;
            int avg_offset = i / c * m_cols + j / c;
            pixels_o[offset].r = (pixval)lpixels[avg_offset].r;
            pixels_o[offset].g = (pixval)lpixels[avg_offset].g;
            pixels_o[offset].b = (pixval)lpixels[avg_offset].b;
        }
    }

    // average value of rgb will be stor in a pixel
    // value of average color / area should range from 0-255 if the
    // image format is correct, explicit casting make this more readable
    pixel average_color = {
        (pixval)(average_r / area),
        (pixval)(average_g / area),
        (pixval)(average_b / area)
    };
    duration = omp_get_wtime() - begin;

    // clean up
    free_lpixel(lpixels);

    printf("CPU mode execution time took %.3lf ms\n", 1000 * duration);
    printf("CPU Average image colour red = %d, green = %d, blue = %d\n", average_color.r, average_color.g, average_color.b);
}
