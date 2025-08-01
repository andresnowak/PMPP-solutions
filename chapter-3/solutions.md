
## Exercises

### 1. Matrix Addition

A **matrix addition** takes two input matrices **A** and **B** and produces one output matrix **C**. Each element of the output matrix **C** is the sum of the corresponding elements of the input matrices **A** and **B**, i.e.,

\[
C[i][j] = A[i][j] + B[i][j].
\]

For simplicity, we will only handle square matrices whose elements are single-precision floating-point numbers. Write a matrix-addition kernel and the host stub function that can be called with four parameters:

- pointer to the output matrix,
- pointer to the first input matrix,
- pointer to the second input matrix,
- number of elements in each dimension.

Follow the instructions below:

A. Write the **host stub function** by allocating memory for the input and output matrices, transferring input data to device; launch the kernel, transferring the output data to host and freeing the device memory for the input and output data. Leave the execution configuration parameters open for this step.

B. Write a **kernel** that has each thread produce **one output matrix element**. Fill in the execution configuration parameters for this design.

C. Write a **kernel** that has each thread produce **one output matrix row**. Fill in the execution configuration parameters for the design.

D. Write a **kernel** that has each thread produce **one output matrix column**. Fill in the execution configuration parameters for the design.

E. Analyze the **pros and cons** of each kernel design above.

**Answer**:
- B
```c++
__global__ void matrix_addition(float *A, float *B, float *C, const int N, const int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // position in x
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // position in y

    if (idx < M && idy < N) {
        int position = idy * M + idx;
        C[position] = A[position] + B[position];
    } 
}
```
- C
```c++
__global__ void matrix_addition_2(float *A, float *B, float *C, const int N, const int M) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // position in y

    if (idy < N) {
        for (int idx = 0; idx < M; ++idx) {
            int position = idy * M + idx;
            C[position] = A[position] + B[position]; 
        }
    } 
}
```   
- D
```c++
__global__ void matrix_addition_3(float *A, float *B, float *C, const int N, const int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // position in x

    if (idx < M) {
        for (int idy = 0; idy < N; ++idy) {
            int position = idy * M + idx;
            C[position] = A[position] + B[position];
        }
    } 
}
```
- E
  - For answer B, this is the one that has perfect load balance, and has coalesced memory access (row-major)
  - For answer C, it has Coalesced memory access along rows, but its cons is that it has bad occupancy for small N, like here we only use 64 threads and we only have 2 blocks (vectors) and has more control-flow overhead (because of the for loop)
  - For answer D, its cons are that it doesn't have coalesced memory access (because it is accessing memory in column-major layout when we have row-major layout) and it has bad occupancy for small M and has more control-flow overhead (because of the for loop).

---

### 2. Matrix–Vector Multiplication

A **matrix–vector multiplication** takes an input matrix **B** and a vector **C** and produces one output vector **A**. Each element of the output vector **A** is the dot product of one row of the input matrix **B** and **C**, i.e.,

\[
A[i] = \sum_j B[i][j] \cdot C[j].
\]

For simplicity, we will only handle square matrices whose elements are single-precision floating-point numbers. Write a matrix–vector multiplication kernel and a host stub function that can be called with four parameters:

- pointer to the output vector,
- pointer to the input matrix,
- pointer to the input vector,
- number of elements in each dimension.

Use **one thread to calculate an output vector element**.

---

### 3. Thread Block Configuration

If the SM of a CUDA device can take up to **1536 threads** and up to **4 thread blocks**, which of the following block configurations would result in the **largest number of threads** in the SM?

A. 128 threads per block  
B. 256 threads per block  
C. 512 threads per block  
D. 1024 threads per block  

**Answer**: C

---

### 4. Vector Addition Grid Size

For a vector addition, assume that the vector length is **2000**, each thread calculates one output element, and the thread block size is **512 threads**. How many threads will be in the grid?

A. 2000  
B. 2024  
C. 2048  
D. 2096  

**Answer**: C

---

### 5. Warp Divergence

With reference to the previous question, how many **warps** do you expect to have divergence due to the boundary check on vector length?

A. 1  
B. 2  
C. 3  
D. 6  

**Answer**: A, because the last last warp won't have any divergence as all threads will follow the same path of not doing anything as they are outside of bounds

---

### 6. Image Kernel Dimensions

You need to write a kernel that operates on an image of size **400 × 900** pixels. You would like to assign **one thread to each pixel**. You would like your thread blocks to be **square** and to use the **maximum number of threads per block possible** on the device (your device has compute capability **3.0**). How would you select the **grid dimensions** and **block dimensions** of your kernel?

**Answer**:
- Compute capability **3.0** has 1024 threads per block
- so we can have blocks of size $32 \cdot 32 = 1024$
- The dimensions of block would be $\text{block dim} = {32, 32, 1}$ and for grid would be $\text{grid dim} = {29, 13, 1}$

---

### 7. Idle Threads

With reference to the previous question, how many **idle threads** do you expect to have?

**Answer**: $13 * 32 * 32 * 29 = 386048$ and we have that $400 * 900 = 360000$, so we have $386048-360000=26048$ idle threads

---

### 8. Barrier Synchronization Overhead

Consider a hypothetical block with **8 threads** executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections:

2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9 and spend the rest of their time waiting for the barrier. What **percentage** of the total execution time of the thread is spent **waiting for the barrier**?

**Answer**: 
Time until threads can continue is 3.0 ms as that is the last thread to finish.
Sum of all their waiting times is $S = {2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9}$, $\text{waiting time} = \sum^{|S|}_{i=1} (3.0 - x_i) = 4.1ms$ and total execution time is $\text{execution time} = 3.0ms * |S| = 24.0ms$.
So the percentage of waiting time is $\frac{4.1}{24} \cdot 100\% = 17.1\%$ 

---

### 9. Multiprocessor Assignment Feasibility

Indicate which of the following assignments per multiprocessor is **possible**. In the case where it is **not possible**, indicate the **limiting factor(s)**.

A. 8 blocks with 128 threads each on a device with compute capability **1.0**  
B. 8 blocks with 128 threads each on a device with compute capability **1.2**  
C. 8 blocks with 128 threads each on a device with compute capability **3.0**  
D. 16 blocks with 64 threads each on a device with compute capability **1.0**  
E. 16 blocks with 64 threads each on a device with compute capability **1.2**  
F. 16 blocks with 64 threads each on a device with compute capability **3.0**

**Answer**:
- Correct ones are B, C and F
- Incorrect ones are A, because it can only have maximum of 512 threads (or 768 threads not sure), but here we get $8 \cdot 128 = 1024 > 512$. then for D and E is incorrect because they can only have 8 blocks while compute capability can have 16 blocks in an SM

---

### 10. Omitting `__syncthreads()`

A CUDA programmer says that if they launch a kernel with **only 32 threads in each block**, they can **leave out the `__syncthreads()` instruction** wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

**Answer**: No, because even though a warps executes in lockstep, threads in a warp can have different execution timings, if all threads in a warp need to complete a phase before they can move on, it is necessary to barrier synchronization like `__syncwarp()` (but this is only if warps have control divergence, if not it is possible to do inter thread communication with things like `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`)

---

### 11. Tiled Matrix Multiplication

A student mentioned that he was able to multiply two **1024 × 1024** matrices by using a **tiled matrix multiplication** code with **32 × 32 thread blocks**. He is using a CUDA device that allows up to **512 threads per block** and up to **8 blocks per SM**. He further mentioned that **each thread in a thread block calculates one element of the result matrix**. What would be your reaction and why?

**Answer**: That is not possible because $32 \cdot 32 = 1024$ and the blocks can only have 512 threads