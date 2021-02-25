/**
 * @file libppm.c
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
#include <stdlib.h>

#include "libppm.h"

/**
 * @Read a character from the file stream and skip comments
 *  No EOF check here, this will be done after calling the function
 *
 * @param fp
 * @return char
 */
char ppm_getc(FILE *fp) {
    int ch = 0;

    // skip comments (start with #)
    while ((ch = getc(fp)) == '#') {
        while ((ch = getc(fp)) != '\n')
            ;
    }

    return (char)ch;
}

/**
 * @Read an integer from the file stream
 *  For reading magic number, width, height, max
 *  color value and rgb values in plain text
 *
 * @param fp
 * @return int
 */
int ppm_getint(FILE *fp) {
    char ch = 0;
    int i = 0;

    // keep reading a character until it is not whitespace
    // do while loop is suitable for checking ch after calling ppm_getc()
    // or this can be done by while(isspace(ch = ppm_getc(fp))) with ctype.h
    do {
        ch = ppm_getc(fp);
    } while (ch == ' ' || ch == '\t' || ch == '\n');

    // EOF check and number check
    if (ch == EOF) {
        fprintf(stderr, "Error: EOF detected during reading.\n");
        exit(1);
    }
    if (ch < '0' || ch > '9') {
        fprintf(stderr, "Error: Unknow data detected when parsing pixels.\n");
        exit(1);
    }

    // convert string to integer
    // above check ensure ch is a digit, so do first and check the next
    do {
        i = i * 10 + ch - '0';
        ch = ppm_getc(fp);
    } while (ch >= '0' && ch <= '9');

    return i;
}

/**
 * @Read magic number
 *  call ppm_getc() twice for P and 3 (or 6)
 *
 * @param fp
 * @return PPM_FORMAT
 */
PPM_FORMAT ppm_readmagicnumber(FILE *fp) {
    // check P
    if ((ppm_getc(fp)) != 'P') {
        fprintf(stderr, "Error: Incorrect format for PPM, the first line should be P3 or P6.\n");
        exit(1);
    }

    // check binary or plain text (3 or 6)
    int n = ppm_getint(fp);
    if (n == 3) {
        return PPM_PLAIN_TEXT;
    }
    if (n == 6) {
        return PPM_BINARY;
    }

    fprintf(stderr, "Error: Incorrect format for PPM, the first line should be P3 or P6.\n");
    exit(1);
}

/**
 * @Memory allocation of pixel array
 *
 * @param cols
 * @param rows
 * @return pixel*
 */
pixel *ppm_allocarray(int cols, int rows) {
    pixel *pixels = NULL;

    pixels = (pixel *)malloc(rows * cols * sizeof(pixel));
    if (pixels == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(1);
    }

    return pixels;
}

/**
 * @Free pixel array
 *
 * @param pixels
 */
void ppm_freearray(pixel *pixels) {
    free(pixels);
}

/**
 * @Read ppm header and check whether value is valid
 *
 * @param fp
 * @param colsP
 * @param rowsP
 * @param maxvalP
 * @param formatP
 */
void ppm_readppminit(FILE *fp, int *colsP, int *rowsP, pixval *maxvalP, PPM_FORMAT *formatP) {
    *formatP = ppm_readmagicnumber(fp);
    if ((*colsP = ppm_getint(fp)) <= 0) {
        fprintf(stderr, "Error: Image width error, incorrect format for ppm.\n");
        exit(1);
    }
    if ((*rowsP = ppm_getint(fp)) <= 0) {
        fprintf(stderr, "Error: Image height error, incorrect format for ppm.\n");
        exit(1);
    }
    // read the max color value, must be 255 here
    if ((*maxvalP = (pixval)ppm_getint(fp)) != 255) {
        fprintf(stderr, "Error: Input PPM maximum colour value of %d found. Only 255 (8-bits per channel) is supported.\n", *maxvalP);
        exit(1);
    }
}

/**
 * @Read an entire image
 *
 * @param fp
 * @param colsP
 * @param rowsP
 * @param maxvalP
 * @return pixel*
 */
pixel *ppm_readppm(FILE *fp, int *colsP, int *rowsP, pixval *maxvalP) {
    PPM_FORMAT format;
    pixel *pixels;

    ppm_readppminit(fp, colsP, rowsP, maxvalP, &format);
    pixels = ppm_allocarray(*colsP, *rowsP);

    int n_pixels = (*rowsP) * (*colsP);

    if (format == PPM_BINARY) {
        fread(pixels, sizeof(pixel), n_pixels, fp);
        return pixels;
    }
    if (format == PPM_PLAIN_TEXT) {
        for (int i = 0; i < n_pixels; ++i) {
            pixels[i].r = ppm_getint(fp);
            pixels[i].g = ppm_getint(fp);
            pixels[i].b = ppm_getint(fp);
        }
        return pixels;
    }

    fprintf(stderr, "Error: Incorrect format for PPM, the first line should be P3 or P6.\n");
    exit(1);
}

/**
 * @Write an entire image
 *
 * @param fp
 * @param pixels
 * @param cols
 * @param rows
 * @param maxvalP
 * @param forceplain
 */
void ppm_writeppm(FILE *fp, pixel *pixels, int cols, int rows, pixval maxvalP, int forceplain) {
    if (!forceplain) {
        fprintf(fp, "P6\n%d\n%d\n%d\n", cols, rows, maxvalP);
        fwrite(pixels, sizeof(pixel), rows * cols, fp);
        return;
    }

    fprintf(fp, "P3\n%d\n%d\n%d", cols, rows, maxvalP);
    for (int i = 0; i < rows; ++i) {
        int index = i * cols;
        fprintf(fp, "\n%d %d %d", pixels[index].r, pixels[index].g, pixels[index].b);
        for (int j = 1; j < cols; ++j) {
            ++index;
            fprintf(fp, "\t%d %d %d", pixels[index].r, pixels[index].g, pixels[index].b);
        }
    }
}
