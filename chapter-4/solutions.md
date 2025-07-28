Consider the following CUDA kernel and the corresponding host function that calls it:

### Exercises

#### 1. Warp-related questions

```c++
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if(i % 2 == 0) {
        a[i] = b[i] * 2;
    }
    for(unsigned int j = 0; j < 5 - (i % 3); ++j) {
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<<(N + 128 - 1) / 128, 128>>>(a_d, b_d);
}
```
a. **What is the number of warps per block?** $128 / 32 = 4$
b. **What is the number of warps in the grid?** $1024/128=8$ and $8 * 4 = 32 \text{warps}$
c. **For the statement on line 04:**  
   i. How many warps in the grid are **active**? 3 warps in a block and we have 8 blocks so 24 warps
   ii. How many warps in the grid are **divergent**? 2 warps in a block and we have 8 blocks so 16 warps
   iii. What is the **SIMD efficiency (%)** of warp 0 of block 0? 100%
   iv. What is the **SIMD efficiency (%)** of warp 1 of block 0? 25%
   v. What is the **SIMD efficiency (%)** of warp 3 of block 0? 75% (the 3rd warp not the second one)

d. **For the statement on line 07:**  
   i. How many warps in the grid are **active**? 32 
   ii. How many warps in the grid are **divergent**? 32  
   iii. What is the **SIMD efficiency (%)** of warp 0 of block 0? 50%  

e. **For the loop on line 09:**  
   i. How many iterations have **no divergence**? 3 iterations. Because $5 - (i \% 3)$, means we have $5 - (0 \% 3) = 5, 5 - (1 \% 3) = 4, 5-(2 \% 3) = 3$, so in total only 3 iterations will see no divergence, (because some threads will only execute 3 iterations)
   ii. How many iterations have **divergence**? 2 iterations 

---

#### 2. Vector Addition
For a vector addition, assume that the vector length is **2000**, each thread calculates one output element, and the thread block size is **512 threads**.  
**How many threads will be in the grid?**

---

#### 3. Warp Divergence in Vector Addition
For the previous question, **how many warps do you expect to have divergence due to the boundary check on vector length?**

---

#### 4. Barrier Synchronization
Consider a hypothetical block with **8 threads** executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections:  
`2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9`  
They spend the rest of their time waiting for the barrier.  
**What percentage of the threads’ total execution time is spent waiting for the barrier?**

---

#### 5. `__syncthreads()` Omission
A CUDA programmer says that if they launch a kernel with **only 32 threads in each block**, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed.  
**Do you think this is a good idea? Explain.**

---

#### 6. Block Configuration for Maximum Threads
If a CUDA device’s SM can take up to **1536 threads** and up to **4 thread blocks**, which of the following block configurations would result in the **most number of threads in the SM**?

a. 128 threads per block  
b. 256 threads per block  
c. 512 threads per block  
d. 1024 threads per block  

---

#### 7. SM Assignment Feasibility
Assume a device that allows up to **64 blocks per SM** and **2048 threads per SM**. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the **occupancy level**.

a. 8 blocks with 128 threads each  
b. 16 blocks with 64 threads each  
c. 32 blocks with 32 threads each  
d. 64 blocks with 32 threads each  
e. 32 blocks with 64 threads each  

---

#### 8. Kernel Occupancy Check
Consider a GPU with the following hardware limits:  
- **2048 threads per SM**  
- **32 blocks per SM**  
- **64K (65,536) registers per SM**  

For each of the following kernel characteristics, specify whether the kernel can achieve **full occupancy**. If not, specify the **limiting factor**.

a. The kernel uses **128 threads per block** and **30 registers per thread**.  
b. The kernel uses **32 threads per block** and **29 registers per thread**.  
c. The kernel uses **256 threads per block** and **34 registers per thread**.  

---

#### 9. Matrix Multiplication Feasibility
A student mentions that they were able to multiply two **1024 × 1024 matrices** using a matrix multiplication kernel with **32 × 32 thread blocks**. The student is using a CUDA device that allows up to **512 threads per block** and up to **8 blocks per SM**. The student further mentions that each thread in a thread block calculates one element of the result matrix.  
**What would be your reaction and why?**