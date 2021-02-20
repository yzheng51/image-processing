/**
 * @file libppm.h
 * @yzheng
 * @read and write ppm file
 *
 * @version 0.1
 * @date 2019-03-08
 *
 * References
 *
 * [1] Jef Poskanzer., (1988). URT: pbmplus/pbm/libpbm4.c File Reference [online]. doxygen documentation | Fossies Dox. [Viewed 08 Mar 2019].
 *     Available from: https://fossies.org/dox/URT3.1a/libpbm4_8c.html
 *
 * [2] Tony Hansen and Jef Poskanzer., (1989, 1991). User manual for old ppm functions [online]. User manual for old ppm functions. [Viewed 08 Mar 2019].
 *     Available from: http://netpbm.sourceforge.net/doc/libppm.html
 *
 * @copyright Copyright (c) 2019, yzheng
 *
 */
#ifndef __LIBPPM_H__
#define __LIBPPM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

// enum for ppm format
typedef enum PPM_FORMAT { PPM_BINARY, PPM_PLAIN_TEXT } PPM_FORMAT;

// data type for color and rgb structure
typedef unsigned char pixval;
typedef struct _pixel {
    pixval r, g, b;
} pixel;

// read character, byte, integer or ppm format (P3 or P6) from ppm file
// part of source code of below three functions are from Jef Poskanzer (1988)
// I change some code based on my understanding and the task
char ppm_getc(FILE *fp);
int ppm_getint(FILE *fp);
PPM_FORMAT ppm_readmagicnumber(FILE *fp);

/* Below function prototypes are from http://netpbm.sourceforge.net/doc/libppm.html */

// allocate and free memory for pixels
pixel *ppm_allocarray(int cols, int rows);
void ppm_freearray(pixel *pixels);

// read and write ppm file
pixel *ppm_readppm(FILE *fp, int *colsP, int *rowsP, pixval *maxvalP);
void ppm_writeppm(FILE *fp, pixel *pixels, int cols, int rows, pixval maxvalP, int forceplain);

#ifdef __cplusplus
}
#endif

#endif