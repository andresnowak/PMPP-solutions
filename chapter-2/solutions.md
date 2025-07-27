## Exercises

1. If we want to use each thread to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to data index?  
   A. `i = threadIdx.x + threadIdx.y;`  
   B. `i = blockIdx.x + threadIdx.x;`  
   C. `i = blockIdx.x * blockDim.x + threadIdx.x;`  
   D. `i = blockIdx.x * threadIdx.x;`

**Answer**: C

2. Assume that we want to use each thread to calculate two (adjacent) elements of a vector addition. What would be the expression for mapping the thread/block indices to `i`, the data index of the first element to be processed by a thread?  
   A. `i = blockIdx.x * blockDim.x + threadIdx.x + 2;`  
   B. `i = blockIdx.x * threadIdx.x * 2;`  
   C. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`  
   D. `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**Answer**: C, Because in the end a threadgroup (a block) processes two blocks of this vector addition

3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2 * blockDim.x` consecutive elements that form two sections. All threads in each block will first process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable `i` should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?  
   A. `i = blockIdx.x * blockDim.x + threadIdx.x + 2;`  
   B. `i = blockIdx.x * threadIdx.x * 2;`  
   C. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`  
   D. `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**Answer**: D

4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel launch to have a minimal number of thread blocks to cover all output elements. How many threads will be in the grid?  
   A. 8000  
   B. 8196  
   C. 8192  
   D. 8200

**Answer**: C, as $\lceil \frac{8000}{1024} \rceil = 8$ and $8 * 1024 = 8092$ 

5. If we want to allocate an array of `v` integer elements in CUDA device global memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?  
   A. `n`  
   B. `v`  
   C. `n * sizeof(int)`  
   D. `v * sizeof(int)`

**Answer**: D

6. If we want to allocate an array of `n` floating-point elements and have a floating-point pointer variable `d_A` to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc()` call?  
   A. `n`  
   B. `(void *) d_A`  
   C. `*d_A`  
   D. `(void **) &d_A`

**Answer**: D

7. If we want to copy 3000 bytes of data from host array `h_A` (h_A is a pointer to element 0 of the source array) to device array `d_A` (d_A is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?  
   A. `cudaMemcpy(3000, h_A, d_A, cudaMemcpyHostToDevice);`  
   B. `cudaMemcpy(h_A, d_A, 3000, cudaMemcpyDeviceToHost);`  
   C. `cudaMemcpy(d_A, h_A, 3000, cudaMemcpyHostToDevice);`  
   D. `cudaMemcpy(3000, d_A, h_A, cudaMemcpyHostToDevice);`

**Answer**: C

8. How would one declare a variable `err` that can appropriately receive returned value of a CUDA API call?  
   A. `int err;`  
   B. `cudaError err;`  
   C. `cudaError_t err;`  
   D. `cudaSuccess_t err;`

**Answer**: C

9. A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious: he had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?

**Answer**: He instead can use both `__device__` and `__host__` keywords so the same kernel can run on CPU and GPU (can be called by cpu (host) or by gpu (device)) 