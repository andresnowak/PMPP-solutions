#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matmul(const float *input_a, const float *input_b, float *output_c, size_t m, size_t n, size_t k)
{
    // We will repeat the work in N dimension so in B
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];

    int col_start = blockIdx.x * TILE_WIDTH * COARSE_FACTOR + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float p_value[COARSE_FACTOR];
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c)
    {
        p_value[c] = 0;
    }

    for (int pk = 0; pk < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++pk)
    {
        int a_col = TILE_WIDTH * pk + threadIdx.x;
        if (row < m && a_col < k)
        {
            Mds[threadIdx.y][threadIdx.x] = input_a[row * k + a_col];
        }
        else
        {
            Mds[threadIdx.y][threadIdx.x] = 0;
        }


        for (int c = 0; c < COARSE_FACTOR; ++c)
        {

            int b_row = (threadIdx.y + TILE_WIDTH * pk);
            int b_col = col_start + TILE_WIDTH * c;
            if (b_row < k && b_col < n)
            {
                Nds[threadIdx.y][threadIdx.x] = input_b[b_row * n + b_col];
            }
            else
            {
                Nds[threadIdx.y][threadIdx.x] = 0;
            }

            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; ++i)
            {
                p_value[c] += Mds[threadIdx.y][i] * Nds[i][threadIdx.x]; // <row, column>
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c)
    {
        int col = col_start + TILE_WIDTH * c;
        if (row < m && col < n)
        {
            output_c[row * n + col] = p_value[c];
        }
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

    // Create the input matrix and vector
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            A[i * K + j] = static_cast<float>(3);
        }
    }
    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B[i * N + j] = static_cast<float>(2);
        }
    }

    // Copy to device
    cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid_dim((N + TILE_WIDTH * COARSE_FACTOR - 1) / (TILE_WIDTH * COARSE_FACTOR), (M + TILE_WIDTH - 1) / TILE_WIDTH, 1); // Because now each block does tiles * COARSE_FACTOR of work in the N dimension of B for the output of C
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