#include <cuda_runtime.h>
#include <stdio.h>

#define FILTER_RADIUS 7                                                  // Biggest one we will use
__constant__ float G[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)]; // We create it 1D so as to have it bigger than our real kernel, and not deal with offsets
// The constant (__constant__) memory variables have to be declared outside any function as it tells the compile to put the variable in the consant "cache" memory of the device

__global__ void convolution(const float *f, float *o, const int width, const int height, const int radius_x, const int radius_y)
{
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_row >= height || out_col >= width)
        return;

    float p_value = 0.0f;

    for (int i = 0; i < (2 * radius_y + 1); i++)
    {
        for (int j = 0; j < (2 * radius_x + 1); j++)
        {
            int in_row = out_row - radius_y + i;
            int in_col = out_col - radius_x + j;

            if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
            {
                p_value += f[in_row * width + in_col] * G[i * (2 * radius_x + 1) + j];
            }
        }
    }

    o[out_row * width + out_col] = p_value;
}

int main()
{
    int n_x = 1 << 5;
    int n_y = 1 << 5;

    int radius_x = 2;
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


    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3((n_x + threadsPerBlock.x - 1) / threadsPerBlock.x, (n_y + threadsPerBlock.y - 1) / threadsPerBlock.y);

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