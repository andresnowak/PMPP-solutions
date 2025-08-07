#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 32

__global__ void matmul(const float *input_a, const float *input_b, float *output_c, size_t m, size_t n, size_t k)
{
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    float p_value = 0;
    for (int pk = 0; pk < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++pk)
    {
        int a_col = TILE_WIDTH * pk + threadIdx.x;
        if (py < m && a_col < k)
        {
            Nds[threadIdx.y][threadIdx.x] = input_a[py * k + a_col];
        }
        else
        {
            Nds[threadIdx.y][threadIdx.x] = 0;
        }
        int b_row = (threadIdx.y + TILE_WIDTH * pk);
        if (b_row < k && px < n)
        {
            Mds[threadIdx.y][threadIdx.x] = input_b[px * k + b_row];
            // Here we are transposing b to get what we had before corner turning (because now b is transposes we have to get the data in in column order instead of row order but then we save the data in the shared memory as row order)
        }
        else
        {
            Mds[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i)
        {
            p_value += Nds[threadIdx.y][i] * Mds[i][threadIdx.x]; // <row, column>
        }

        __syncthreads();
    }

    if (py < m && px < n)
    {
        output_c[py * n + px] = p_value;
    }
}

void print_matrix(float *A, const int M, const int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << A[i * N + j] << ",";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main()
{
    float *A;
    float *A_d;
    float *B;
    float *B_d;
    float *C;
    float *C_d;

    const int N = 128;
    const int M = 128;
    const int K = 128;

    // Allocate host and device memory
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));
    cudaMalloc(&A_d, M * K * sizeof(float));
    cudaMalloc(&B_d, K * N * sizeof(float));
    cudaMalloc(&C_d, M * N * sizeof(float));

    // Create the input matrix and vector (Now we say that memory is saved in Column major order)
    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            A[i * K + j] = static_cast<float>(3);
        }
    }
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            B[i * K + j] = static_cast<float>(2);
        }
    }

    // Copy to device
    cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid_dim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    matmul<<<grid_dim, block_dim>>>(A_d, B_d, C_d, M, N, K);

    // Copy result back to host
    cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix(C, M, N);
    std::cout << "\n\n";

    // Cleanup
    free(A);
    free(B);
    free(C);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}