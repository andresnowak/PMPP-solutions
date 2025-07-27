#include <iostream>
#include <cuda_runtime.h>


__global__ void matrix_vector_mul(float *A, float *B, float *C, const int N, const int M) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // position in y

    if (idy < N) {
        float sum = 0;
        for (int idx = 0; idx < M; ++idx) {
            sum += A[idy * M + idx] * B[idx];
        }
        C[idy] = sum;
    } 
}


void print_vector(float *A, const int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << A[i] << ",";
    }
    std::cout << "\n";
}


int main() {
    float* A;
    float* A_d;
    float* B;
    float* B_d;
    float* C; 
    float* C_d;

    const int N = 32;
    const int M = 32;

    // Allocate host and device memory
    A = (float*)malloc(N * M * sizeof(float));
    B = (float*)malloc(M * sizeof(float));
    C = (float*)malloc(N * sizeof(float));
    cudaMalloc(&A_d, N * M * sizeof(float));
    cudaMalloc(&B_d, M * sizeof(float));
    cudaMalloc(&C_d, N * sizeof(float));
    
    // Create the input matrix and vector
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i * M + j] = static_cast<float>(3);
            B[j] = static_cast<float>(2); // I know we are assigning the value at the same pos multiple times
        }
    }

    // Copy to device
    cudaMemcpy(A_d, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, M * sizeof(float), cudaMemcpyHostToDevice);
   
    dim3 block(1, 32, 1);
    dim3 grid(1, static_cast<int>((static_cast<float>(N) + block.y - 1) / block.y), 1);
    matrix_vector_mul<<<grid, block>>>(A_d, B_d, C_d, N, M);
    
    // Copy result back to host
    cudaMemcpy(C, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    print_vector(C, N);
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