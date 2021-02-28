/**
 * @file mosaic.c
 * @yzheng
 * @mosaic
 * @version 0.1
 * @date 2019-03-08
 *
 * @copyright Copyright (c) 2019, yzheng
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libppm.h"
#include "filter.h"
#include "cuda_filter.h"

#define FAILURE 0
#define SUCCESS !FAILURE

void print_help();
int process_command_line(int argc, char *argv[]);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

unsigned int c = 0;
MODE execution_mode = CPU;
PPM_FORMAT fmt = PPM_BINARY;
char *input_file, *output_file;

int main(int argc, char *argv[]) {
    if (process_command_line(argc, argv) == FAILURE)
        return 1;

    // read input image file (either binary or plain text PPM)
    FILE *fp = NULL;
    int cols, rows;              // width and height of the input ppm file
    pixval maxval;               // max color value for ppm file, which must be 255 and will be checked in ppm_readppm()
    pixel *pixels_i, *pixels_o;  // pixel data of the output ppm file

    if ((fp = fopen(input_file, "rb")) == NULL) {
        fprintf(stderr, "Error: opening '%s' failed. Please check your filename.\n", input_file);
        return FAILURE;
    }
    pixels_i = ppm_readppm(fp, &cols, &rows, &maxval);
    pixels_o = ppm_allocarray(cols, rows);

    if (c > (unsigned int)rows) {
        fprintf(stderr, "Error: Height of the image is less than the specified C value.\n");
        exit(1);
    }
    if (c > (unsigned int)cols) {
        fprintf(stderr, "Error: Width of the image is less than the specified C value.\n");
        exit(1);
    }

    //TODO: execute the mosaic filter based on the mode
    switch (execution_mode){
        case (CPU) : {
            mosaic_transform(pixels_o, pixels_i, cols, rows, c);
            break;
        }
        case (OPENMP) : {
            mosaic_transform_omp(pixels_o, pixels_i, cols, rows, c);
            break;
        }
        case (CUDA) : {
            mosaic_transform_cuda(pixels_o, pixels_i, cols, rows, c);
            break;
        }
        case (ALL) : {
            mosaic_transform(pixels_o, pixels_i, cols, rows, c);
            printf("\n");
            mosaic_transform_omp(pixels_o, pixels_i, cols, rows, c);
            printf("\n");
            mosaic_transform_cuda(pixels_o, pixels_i, cols, rows, c);
            break;
        }
    }

    // close the input file and set the file pointer to null
    // this pointer will be used by output file
    // so it is set to null manually for safety
    fclose(fp); fp = NULL;

    // save the output image file (from last executed mode)
    if ((fp = fopen(output_file, "wb")) == NULL) {
        fprintf(stderr, "Error: opening '%s' failed. Please check your filename.\n", output_file);
        return FAILURE;
    }
    ppm_writeppm(fp, pixels_o, cols, rows, maxval, fmt);

    // clean up
    fclose(fp);
    free(input_file); free(output_file);
    ppm_freearray(pixels_i); ppm_freearray(pixels_o);

    return 0;
}

void print_help() {
    printf("mosaic C M -i input_file -o output_file [options]\n");

    printf("where:\n");
    printf("\tC              Is the mosaic cell size which should be any positive\n"
           "\t               power of 2 number \n");
    printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
           "\t               ALL. The mode specifies which version of the simulation\n"
           "\t               code should execute. ALL should execute each mode in\n"
           "\t               turn.\n");
    printf("\t-i input_file  Specifies an input image file\n");
    printf("\t-o output_file Specifies an output image file which will be used\n"
           "\t               to write the mosaic image\n");
    printf("[options]:\n");
    printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
           "\t               PPM_PLAIN_TEXT\n ");
}

int check_cell_size(char *ch) {
    // the first character in the second argument should be a digit or +
    // after then, they should be digit or '.' (for float number)
    if (!(*ch == '+' || (*ch >= '0' && *ch <= '9'))) {
        fprintf(stderr, "Error: Mosaic cell size argument 'C' must be in greater than 0.\n");
        return FAILURE;
    }

    // ch should be end up with '\0', otherwise it is invalid
    do {
        ch++;
    } while ((*ch >= '0' && *ch <= '9') || *ch == '.');

    if (*ch) {
        fprintf(stderr, "Error: Mosaic cell size argument 'C' must be in greater than 0.\n");
        return FAILURE;
    }

    return SUCCESS;
}

MODE get_mode(char *ch) {
    if (strcmp(ch, "CPU") == 0) {
        return CPU;
    }
    if (strcmp(ch, "OPENMP") == 0) {
        return OPENMP;
    }
    if (strcmp(ch, "CUDA") == 0) {
        return CUDA;
    }
    if (strcmp(ch, "ALL") == 0) {
        return ALL;
    }

    fprintf(stderr, "Error: Mode should be CPU, OPENMP, CUDA or ALL.\n");
    exit(1);
}

int process_command_line(int argc, char *argv[]){
    if (argc < 7){
        fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
        print_help();
        return FAILURE;
    }

    // read in the non optional command line arguments
    if (check_cell_size(argv[1]) == FAILURE) {
        return FAILURE;
    }
    c = (unsigned int)atoi(argv[1]);

    // read in the mode
    execution_mode = get_mode(argv[2]);

    // read in the input image name
    if (strcmp(argv[3], "-i") != 0) {
        fprintf(stderr, "Error: Third argument must be '-i' to specify input image filename. Correct usage is...\n");
        print_help();
        return FAILURE;
    }
    // allocate the memory for file name based on the input
    input_file = (char *)malloc((strlen(argv[4]) + 1) * sizeof(char));
    if (input_file == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return FAILURE;
    }
    // use strcpy rather than strncpy here because it is safe
    // the length of the two string must be the same
    strcpy(input_file, argv[4]);

    // read in the output image name
    if (strcmp(argv[5], "-o") != 0) {
        fprintf(stderr, "Error: Fifth argument must be '-o' to specify output image filename. Correct usage is...\n");
        print_help();
        return FAILURE;
    }
    output_file = (char *)malloc((strlen(argv[6]) + 1) * sizeof(char));
    if (output_file == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return FAILURE;
    }
    strcpy(output_file, argv[6]);

    // read in any optional part 3 arguments
    // start checking until the program find '-f'
    for (int i = 7; i < argc; ++i) {
        if (strcmp(argv[i], "-f") != 0) {
            fprintf(stderr, "Warning: Unrecognised optional argument '%s' ignored.\n", argv[i]);
            continue;
        }
        // '-f' found, check whether next argument exist
        if (i + 1 == argc) {
            fprintf(stderr, "Error: 'PPM_BINARY' or 'PPM_PLAIN_TEXT' format expected after '-f' switch.\n");
            return FAILURE;
        }
        if (strcmp(argv[i + 1], "PPM_PLAIN_TEXT") == 0) {
            fmt = PPM_PLAIN_TEXT;
            i++;  // increase the counter to avoid find PPM_PLAIN_TEXT again, safe due to above check
            continue;
        }
        if (strcmp(argv[i + 1], "PPM_BINARY") == 0) {
            i++;  // increase the counter to avoid find PPM_BINARY again
            continue;
        }
        fprintf(stderr, "Error: 'PPM_BINARY' or 'PPM_PLAIN_TEXT' format expected after '-f' switch.\n");
        return FAILURE;
    }

    return SUCCESS;
}
