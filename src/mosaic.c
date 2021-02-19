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

#define FAILURE 0
#define SUCCESS !FAILURE

void print_help();
int process_command_line(int argc, char *argv[]);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

unsigned int c = 0;
MODE execution_mode = CPU;

int main(int argc, char *argv[]) {
    if (process_command_line(argc, argv) == FAILURE)
        return 1;

    //TODO: read input image file (either binary or plain text PPM)

    //TODO: execute the mosaic filter based on the mode
    switch (execution_mode){
        case (CPU) : {
            //TODO: starting timing here

            //TODO: calculate the average colour value

            // Output the average colour value for the image
            printf("CPU Average image colour red = ???, green = ???, blue = ??? \n");

            //TODO: end timing here
            printf("CPU mode execution time took ??? s and ???ms\n");
            break;
        }
        case (OPENMP) : {
            //TODO: starting timing here

            //TODO: calculate the average colour value

            // Output the average colour value for the image
            printf("OPENMP Average image colour red = ???, green = ???, blue = ??? \n");

            //TODO: end timing here
            printf("OPENMP mode execution time took ??? s and ?? ?ms\n");
            break;
        }
        case (CUDA) : {
            printf("CUDA Implementation not required for assignment part 1\n");
            break;
        }
        case (ALL) : {
            //TODO
            break;
        }
    }

    //save the output image file (from last executed mode)

    return 0;
}

void print_help() {
    printf("mosaic_ C M -i input_file -o output_file [options]\n");

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

int process_command_line(int argc, char *argv[]){
    if (argc < 7){
        fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
        print_help();
        return FAILURE;
    }

    //first argument is always the executable name

    //read in the non optional command line arguments
    c = (unsigned int)atoi(argv[1]);

    //TODO: read in the mode

    //TODO: read in the input image name

    //TODO: read in the output image name

    //TODO: read in any optional part 3 arguments

    return SUCCESS;
}
