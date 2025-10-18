#include<cuda_runtime.h>

#include <stdio.h>

#define FILTER_RADIUS 2

    __constant__ float G[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)]; // We create it 1D so as to have it bigger than our real kernel, and not deal with offsets

// The constant (__constant__) memory variables have to be declared outside any function as it tells the compile to put the variable in the consant "cache" memory of the device

#define IN_TILE_DIM 32

#define OUT_TILE_DIM (IN_TILE_DIM - (2 * FILTER_RADIUS + 1) - 1)

__global__ void convolution(const float *f, float *o, const int width, const int height, const int radius_x, const int radius_y)
{

    // the position calculation at the end of the day is still based on the output tile, but the amount of threads we ahve in a block is of size input_tile

    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS; // Substract R as input_tile size is based on 2*R (diameter), but we only want radius for position

    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float input_tile[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < height && col >= 0 && col < width)
    {

        input_tile[threadIdx.y][threadIdx.x] = f[row * width + col];
    }
    else
    {

        input_tile[threadIdx.y][threadIdx.x] = 0; // Ghost cell (default value 0)

        // No memory access
    }

    __syncthreads();

    // output elements

    int tile_col = threadIdx.x - FILTER_RADIUS;

    int tile_row = threadIdx.y - FILTER_RADIUS;

    // turning off the threads at the edges of the block

    if (col >= 0 && col < width && row >= 0 && row < height)
    {

        float p_value = 0.0f;

        // Here we mask based on the output_tile, because we only want to calcualate the real output threads (not the overlapping ones with other blocks, so not the halo threads)

        if (tile_col >= 0 && tile_col < OUT_TILE_DIM && tile_row >= 0 && tile_row < OUT_TILE_DIM)
        {

            for (int i = 0; i < (2 * radius_y + 1); i++)
            {

                for (int j = 0; j < (2 * radius_x + 1); j++)
                {

                    p_value += input_tile[tile_row + i][tile_col + j] * G[i * (2 * radius_x + 1) + j];
                }
            }

            o[row * width + col] = p_value;
        }
    }
}

int main()
{

    int n_x = 1 << 5;

    int n_y = 1 << 5;

    int radius_x = 2; // Same as FILTER_RADIUS

    int radius_y = 2;

    int kernel_width = (2 * radius_x + 1);

    int kernel_height = (2 * radius_y + 1);

    float *f = (float *)malloc(n_x * n_y * sizeof(float));

    float *g = (float *)malloc(kernel_width * kernel_height * sizeof(float));

    float *o = (float *)malloc(n_x * n_y * sizeof(float));

    for (int i = 0; i < n_x * n_y; i++)
    {

        f[i] = 1.0f;
    }

    for (int i = 0; i < kernel_width * kernel_height; i++)
    {

        g[i] = 2.0f;
    }

    float *d_f, *d_o;

    // float *d_g;

    cudaMalloc(&d_f, n_x * n_y * sizeof(float));

    // cudaMalloc(&d_g, kernel_width * kernel_height * sizeof(float));

    cudaMalloc(&d_o, n_x * n_y * sizeof(float));

    cudaMemcpy(d_f, f, n_x * n_y * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t cpy = cudaMemcpyToSymbol(G, g, kernel_width * kernel_height * sizeof(float), 0, cudaMemcpyHostToDevice); // This is a special copy function that tell CUDA this data will not be changed during kernel execution

    if (cpy != cudaSuccess)
    {

        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cpy));

        exit(1);
    }

    dim3 threadsPerBlock = dim3(IN_TILE_DIM, IN_TILE_DIM);

    dim3 blocksPerGrid = dim3((n_x + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (n_y + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    convolution<<<blocksPerGrid, threadsPerBlock>>>(d_f, d_o, n_x, n_y, radius_x, radius_y);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {

        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(o, d_o, n_x * n_y * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; i++)
    {

        printf("(%f - %f), ", f[i], o[i]);
    }

    // Cleanup

    cudaFree(d_f);

    // cudaFree(d_g);

    cudaFree(d_o);

    free(f);

    free(g);

    free(o);

    return 0;
}