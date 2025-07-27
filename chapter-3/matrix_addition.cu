#include <iostream>
#include <cuda_runtime.h>


__global__ void matrix_addition(float *A, float *B, float *C, const int N, const int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // position in x
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // position in y

    if (idx < M && idy < N) {
        int position = idy * M + idx;
        C[position] = A[position] + B[position];
    } 
}


__global__ void matrix_addition_2(float *A, float *B, float *C, const int N, const int M) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // position in y

    if (idy < N) {
        for (int idx = 0; idx < M; ++idx) {
            int position = idy * M + idx;
            C[position] = A[position] + B[position]; 
        }
    } 
}

__global__ void matrix_addition_3(float *A, float *B, float *C, const int N, const int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // position in x

    if (idx < M) {
        for (int idy = 0; idy < N; ++idy) {
            int position = idy * M + idx;
            C[position] = A[position] + B[position];
        }
    } 
}

void print_matrix(float *A, const int N, const int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << A[i * M + j] << ",";
        }
        std::cout << "\n";
    }
}


int main() {
    float* A;
    float* A_d;
    float* B;
    float* B_d;
    float* C; 
    float* C_d;

    const int N = 64;
    const int M = 65;

    // Allocate host and device memory
    A = (float*)malloc(N * M * sizeof(float));
    B = (float*)malloc(N * M * sizeof(float));
    C = (float*)malloc(N * M * sizeof(float));
    cudaMalloc(&A_d, N * M * sizeof(float));
    cudaMalloc(&B_d, N * M * sizeof(float));
    cudaMalloc(&C_d, N * M * sizeof(float));
    
    // Create the input matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i * M + j] = static_cast<float>(1);
            B[i * M + j] = static_cast<float>(2);
        }
    }

    // Copy to device
    cudaMemcpy(A_d, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * M * sizeof(float), cudaMemcpyHostToDevice);
   
    dim3 block(32, 32, 1);
    dim3 grid(static_cast<int>(std::ceil(static_cast<float>(M) / block.x)), static_cast<int>(std::ceil(static_cast<float>(N) / block.y)), 1);
    matrix_addition<<<grid, block>>>(A_d, B_d, C_d, N, M);
    
    // Copy result back to host
    cudaMemcpy(C, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix(C, N, M);
    std::cout << "\n\n";

    // Now we do matrix addtion version 2 (where a thread produces one row matrix output)
	cudaMemset(C_d, 0, N * M * sizeof(float)); // single call, async

    block = dim3(1, 32, 1);
    grid = dim3(1, static_cast<int>(std::ceil(static_cast<float>(N) / block.y)), 1);
    matrix_addition_2<<<grid, block>>>(A_d, B_d, C_d, N, M);
    
    // Copy result back to host
    cudaMemcpy(C, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix(C, N, M);
    std::cout << "\n\n";

    // Finally we do matrix addition version 3 (where a thread produces one column matrix output)
    cudaMemset(C_d, 0, N * M * sizeof(float));

    block = dim3(32, 1, 1);
    grid = dim3(static_cast<int>(std::ceil(static_cast<float>(M) / block.x)), 1, 1);
    matrix_addition_3<<<grid, block>>>(A_d, B_d, C_d, N, M);
    
    // Copy result back to host
    cudaMemcpy(C, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix(C, N, M);
    
    
    // Cleanup
    free(A);
    free(B);
    free(C);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    return 0;
}