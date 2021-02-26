/**
 * @file cuda_filter.cu
 * @yzheng
 * @CUDA mosaic filter
 * @version 0.1
 * @date 2019-05-07
 *
 * @copyright Copyright (c) 2019, yzheng
 *
 */
#include <stdlib.h>

#include "filter.h"
#include "cuda_filter.h"

#define WARP_SIZE 32
#define MPR 32  // mosaic cells per row in one block
#define MPR_SQUARE (MPR * MPR)

#define MAX_TPB 256  // max thread per block
#define MAX_TPB_SQUARE (MAX_TPB * MAX_TPB)
#define DELTA(X) ((X) / MAX_TPB)  // c / MAX_TPB, to get number of z axis
#define DELTA_SQUARE(X) ((X) * (X) / MAX_TPB_SQUARE)

/**
 * @Memory check
 *
 * @param msg
 */
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * @Convert pixel array to cuda vector
 *
 * @param dest_pixels
 * @param src_pixels
 * @param cols
 * @param rows
 */
void pixel_to_image(uchar3 *dest_pixels, pixel *src_pixels, int cols, int rows) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = j * rows + i;
            int offset = i * cols + j;
            dest_pixels[index].x = src_pixels[offset].r;
            dest_pixels[index].y = src_pixels[offset].g;
            dest_pixels[index].z = src_pixels[offset].b;
        }
    }
}

/**
 * @Convert cuda vector to pixel array
 *
 * @param dest_pixels
 * @param src_pixels
 * @param cols
 * @param rows
 */
void image_to_pixel(pixel *dest_pixels, uchar3 *src_pixels, int cols, int rows) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = j * rows + i;
            int offset = i * cols + j;
            dest_pixels[offset].r = src_pixels[index].x;
            dest_pixels[offset].g = src_pixels[index].y;
            dest_pixels[offset].b = src_pixels[index].z;
        }
    }
}

/**
 * @Mosaic filter
 *
 * @param pixels_o
 * @param pixels_i
 * @param cols
 * @param rows
 * @param c
 */
void mosaic_transform_cuda(pixel *pixels_o, pixel *pixels_i, int cols, int rows, int c) {
    // checking c
    if (!(c & (c - 1)) == 0) {
        fprintf(stderr, "Error: C should be the power of two in CUDA implementation.\n");
        exit(1);
    }
    int cMPR = c * MPR;  // for computing the grid size in third implementation
    int area = cols * rows;
    int grid_cols = (cols % c) ? cols / c + 1 : cols / c;  // deal with partial mosaic
    int grid_rows = (rows % c) ? rows / c + 1 : rows / c;
    int shared_size = c * sizeof(float3);  // the size of shared variable
    int image_size = area * sizeof(uchar3);
    int average_size = sizeof(float3);
    int temp_size = DELTA_SQUARE(c) * sizeof(float3);  // the teamp array used when c > MAX_TPB
    float ms;
    uchar3 *image_i, *image_o, *d_image_i, *d_image_o;
    float3 *average, *d_average, *d_average_temp;
    cudaEvent_t start, stop;

    // modify the size of the shared variable depends on the size
    if (c <= 4 && rows / cMPR > 1 && cols / cMPR > 1) {
        shared_size = MPR_SQUARE * sizeof(float3);
        grid_cols = (cols % cMPR) ? cols / cMPR + 1 : cols / cMPR;
        grid_rows = (rows % cMPR) ? rows / cMPR + 1 : rows / cMPR;
    }

    if (c > MAX_TPB) {
        shared_size = MAX_TPB * sizeof(float3);
    }

    // create timers
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // memory allocation on host
    cudaMallocHost((void **)&image_i, image_size);
    cudaMallocHost((void **)&image_o, image_size);
    cudaMallocHost((void **)&average, average_size);

    // memory allocation on device
    cudaMalloc((void **)&d_image_i, image_size);
    cudaMalloc((void **)&d_image_o, image_size);
    cudaMalloc((void **)&d_average, average_size);
    cudaMalloc((void **)&d_average_temp, temp_size);
    checkCUDAError("CUDA malloc");

    // convert pixel array to cuda vector
    pixel_to_image(image_i, pixels_i, cols, rows);

    // copy image from host to device
    cudaMemcpy(d_image_i, image_i, image_size, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy");

    cudaEventRecord(start);
    // implementation type from top to bottom:
    // 32 x 32 mosaic cells per block
    // c thread per block with z axis
    // c thread per block without z axis
    if (c <= 4 && rows / cMPR > 1 && cols / cMPR > 1) {
        dim3 blocksPerGrid(grid_rows, grid_cols);
        dim3 threadsPerBlock(MPR, MPR);
        mosaic_filter_ccmpb<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_image_i, d_image_o, d_average, cols, rows, c);
    } else if (c > MAX_TPB) {
        dim3 blocksPerGrid(grid_rows, grid_cols, DELTA_SQUARE(c));
        dim3 threadsPerBlock(MAX_TPB, 1);
        mosaic_filter_cpb_z<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_image_i, d_average_temp, cols, rows, c);
        mosaic_out_cpb_z<<<blocksPerGrid, threadsPerBlock>>>(d_average_temp, d_image_o, d_average, cols, rows, c);
    } else {
        dim3 blocksPerGrid(grid_rows, grid_cols);
        dim3 threadsPerBlock(c, 1);
        mosaic_filter_cpb<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_image_i, d_image_o, d_average, cols, rows);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("CUDA kernel");

    // copy output image and thread_average values from device to host
    cudaMemcpy(image_o, d_image_o, image_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(average, d_average, average_size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy");

    // convert cuda vector to pixel array
    image_to_pixel(pixels_o, image_o, cols, rows);

    pixel average_color = {
        (unsigned char)(average->x / area),
        (unsigned char)(average->y / area),
        (unsigned char)(average->z / area)
    };

    // clean up
    cudaFreeHost(image_o); cudaFreeHost(image_i); cudaFreeHost(average);
    cudaFree(d_image_i); cudaFree(d_image_o); cudaFree(d_average); cudaFree(d_average_temp);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("CUDA mode execution time took %.3lf ms\n", ms);
    printf("CUDA Average image colour red = %d, green = %d, blue = %d\n", average_color.r, average_color.g, average_color.b);
}

// -------------------------------- c threads per block --------------------------------------------
/**
 * @Mosaic filter
 *
 * @param image
 * @param image_output
 * @param average
 * @param cols
 * @param rows
 */
__global__ void mosaic_filter_cpb(uchar3 *image, uchar3 *image_output, float3 *average, int cols, int rows) {
    extern __shared__ float3 s_average[];  // sum of rgb values in one block/mosaic cell

    int c = blockDim.x;
    int x = threadIdx.x + blockIdx.x * c;
    int y = threadIdx.y + blockIdx.y * c;

    int m_area;  // size of one mosaic cell
    int mod_cols = cols % c;
    int mod_rows = rows % c;
    float3 thread_average = make_float3(0, 0, 0);  // sum of one row in one block/mosaic cell

    // calculate size for each mosaic cell depends on position (cope with partial mosaic)
    mod_cols = (y < cols - mod_cols) ? c : mod_cols;
    mod_rows = (x < rows - mod_rows) ? c : mod_rows;
    m_area = mod_rows * mod_cols;

    // traverse over one row of one mosaic cell
    if (x < rows && y < cols) {
        for (int j = 0; j < mod_cols; ++j) {
            int y_offset = y + j;
            int offset = x + y_offset * rows;
            thread_average.x += image[offset].x;
            thread_average.y += image[offset].y;
            thread_average.z += image[offset].z;
        }
    }

    // assign the sum of one row to shared variable
    // perform reduction to get the sum for one block/mosaic cell
    s_average[threadIdx.x] = thread_average;
    __syncthreads();

    // if c < 32, using warp level reduction, otherwise using the normal one
    if (c <= WARP_SIZE) {
        for (int stride = c / 2; stride != 0; stride >>= 1) {
            s_average[threadIdx.x].x += __shfl_down_sync(0xffffffff, s_average[threadIdx.x].x, stride);
            s_average[threadIdx.x].y += __shfl_down_sync(0xffffffff, s_average[threadIdx.x].y, stride);
            s_average[threadIdx.x].z += __shfl_down_sync(0xffffffff, s_average[threadIdx.x].z, stride);
        }
    } else {
        for (int stride = c / 2; stride != 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_average[threadIdx.x].x += s_average[threadIdx.x + stride].x;
                s_average[threadIdx.x].y += s_average[threadIdx.x + stride].y;
                s_average[threadIdx.x].z += s_average[threadIdx.x + stride].z;
            }
            __syncthreads();
        }
    }

    // atomic add to the sum of the entire image
    // averaging operation is in the host function
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&(average->x), s_average->x);
        atomicAdd(&(average->y), s_average->y);
        atomicAdd(&(average->z), s_average->z);
    }

    // assign the rgb value to the output image
    thread_average.x = s_average->x / m_area;
    thread_average.y = s_average->y / m_area;
    thread_average.z = s_average->z / m_area;

    // assign back to the output image
    if (x < rows && y < cols) {
        for (int j = 0; j < mod_cols; ++j) {
            int y_offset = y + j;
            int offset = x + y_offset * rows;
            image_output[offset].x = (unsigned char)thread_average.x;
            image_output[offset].y = (unsigned char)thread_average.y;
            image_output[offset].z = (unsigned char)thread_average.z;
        }
    }
}

/**
 * @Mosaic filter with z axis
 *
 * @param image
 * @param image_output
 * @param temp_average
 * @param cols
 * @param rows
 * @param c
 */
__global__ void mosaic_filter_cpb_z(uchar3 *image, float3 *temp_average, int cols, int rows, int c) {
    // almost the same as the above function
    extern __shared__ float3 s_average[];

    int delta = DELTA(c);  // used to mapping the index below
    int x = threadIdx.x + blockIdx.x * c + (blockIdx.z / delta) * MAX_TPB;
    int y = threadIdx.y + blockIdx.y * c + (blockIdx.z % delta) * MAX_TPB;
    int ibx = blockIdx.x + blockIdx.y * gridDim.x;

    int mod_cols = cols % MAX_TPB;
    float3 thread_average = make_float3(0, 0, 0);

    mod_cols = (y < cols - mod_cols) ? MAX_TPB : mod_cols;

    if (x < rows && y < cols) {
        for (int j = 0; j < mod_cols; ++j) {
            int y_offset = y + j;
            int offset = x + y_offset * rows;
            thread_average.x += image[offset].x;
            thread_average.y += image[offset].y;
            thread_average.z += image[offset].z;
        }
    }

    s_average[threadIdx.x] = thread_average;
    __syncthreads();

    // the initialization of stride will use max thread per block
    for (int stride = MAX_TPB / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_average[threadIdx.x].x += s_average[threadIdx.x + stride].x;
            s_average[threadIdx.x].y += s_average[threadIdx.x + stride].y;
            s_average[threadIdx.x].z += s_average[threadIdx.x + stride].z;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&(temp_average[ibx].x), s_average->x);
        atomicAdd(&(temp_average[ibx].y), s_average->y);
        atomicAdd(&(temp_average[ibx].z), s_average->z);
    }
}

/**
 * @Sum up and generate the output of the mosaic_filter_cpb_z()
 *
 * @param temp_average
 * @param image_output
 * @param average
 * @param cols
 * @param rows
 * @param c
 */
__global__ void mosaic_out_cpb_z(float3 *temp_average, uchar3 *image_output, float3 *average, int cols, int rows, int c) {
    // the same mapping in mosaic_filter_cpb_z()
    int delta = DELTA(c);
    int x = threadIdx.x + blockIdx.x * c + (blockIdx.z / delta) * MAX_TPB;
    int y = threadIdx.y + blockIdx.y * c + (blockIdx.z % delta) * MAX_TPB;
    int ibx = blockIdx.x + blockIdx.y * gridDim.x;

    int m_area;
    int mod_cols = cols % c;
    int mod_rows = rows % c;
    float3 m_average = temp_average[ibx];

    mod_cols = (y < cols - mod_cols) ? c : mod_cols;
    mod_rows = (x < rows - mod_rows) ? c : mod_rows;
    m_area = mod_rows * mod_cols;

    // modify it to ensure the partial calculation will not exceed the boundary
    mod_cols = (y < cols - cols % MAX_TPB) ? MAX_TPB : cols % MAX_TPB;

    if (x < rows && y < cols) {
        for (int j = 0; j < mod_cols; ++j) {
            int y_offset = y + j;
            int offset = x + y_offset * rows;
            image_output[offset].x = (unsigned char)(m_average.x / m_area);
            image_output[offset].y = (unsigned char)(m_average.y / m_area);
            image_output[offset].z = (unsigned char)(m_average.z / m_area);
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.z == 0) {
        atomicAdd(&(average->x), m_average.x);
        atomicAdd(&(average->y), m_average.y);
        atomicAdd(&(average->z), m_average.z);
    }
}

// ---------------------------- 32 x 32 mosaic cells per block ---------------------------------

/**
 * @Mosaic filter (32 x 32 mosaic cells per block)
 *
 * @param image
 * @param image_output
 * @param average
 * @param cols
 * @param rows
 * @param c
 */
__global__ void mosaic_filter_ccmpb(uchar3 *image, uchar3 *image_output, float3 *average, int cols, int rows, int c) {
    extern __shared__ float3 s_average[];

    // mapping the index to position
    int x = (threadIdx.x + blockIdx.x * MPR) * c;
    int y = (threadIdx.y + blockIdx.y * MPR) * c;
    int itx = threadIdx.x + threadIdx.y * MPR;

    int m_area;
    int mod_cols = cols % c;
    int mod_rows = rows % c;

    float3 m_average = make_float3(0, 0, 0);

    // calculation the size to deal with partial mosaic
    mod_cols = (y < cols - mod_cols) ? c : mod_cols;
    mod_rows = (x < rows - mod_rows) ? c : mod_rows;
    m_area = mod_rows * mod_cols;

    // using for loop to sum up the RGB to the register
    if (x < rows && y < cols) {
        for (int i = 0; i < c; ++i) {
            for (int j = 0; j < c; ++j) {
                int x_offset = x + i;
                int y_offset = y + j;
                int offset = x_offset + y_offset * rows;

                m_average.x += image[offset].x;
                m_average.y += image[offset].y;
                m_average.z += image[offset].z;
            }
        }
    }

    // do reduction with shared variable
    s_average[itx] = m_average;
    __syncthreads();

    for (int stride = MPR_SQUARE / 2; stride != 0; stride >>= 1) {
        if (itx < stride) {
            s_average[itx].x += s_average[itx + stride].x;
            s_average[itx].y += s_average[itx + stride].y;
            s_average[itx].z += s_average[itx + stride].z;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&(average->x), s_average[itx].x);
        atomicAdd(&(average->y), s_average[itx].y);
        atomicAdd(&(average->z), s_average[itx].z);
    }

    if (x < rows && y < cols) {
        for (int i = 0; i < c; ++i) {
            for (int j = 0; j < c; ++j) {
                int x_offset = x + i;
                int y_offset = y + j;
                int offset = x_offset + y_offset * rows;

                image_output[offset].x = (unsigned char)(m_average.x / m_area);
                image_output[offset].y = (unsigned char)(m_average.y / m_area);
                image_output[offset].z = (unsigned char)(m_average.z / m_area);
            }
        }
    }
}
