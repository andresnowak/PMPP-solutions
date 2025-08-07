1. Write a matrix-multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.

**Answer**:
```c++
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
```

2. For tiled matrix multiplication, of the possible range of values for `BLOCK_SIZE`, for what values of `BLOCK_SIZE` will the kernel completely avoid uncoalesced accesses to global memory? (You need to consider only square blocks.)
**Answer**: The safe sizes are those that are multiples of warp size (so 32, 64, etc..). The reason is that if we want the values in a warp to come from the same load burst from global memory, we can't have some of the threads in the first row for example and the others in the second row, if we have that imagine if the tile is at the beginning of the matrix, then we would have a very big strided jump from one row to the other, so it is very probable that they will come from different row wires from global memory


3. Consider the following CUDA kernel:  
   ```c++
   __global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ float a_s[256];
        __shared__ float bc_s[4 * 256];

        a_s[threadIdx.x] = a[i];
        for (unsigned int j = 0; j < 4; ++j) {
            bc_s[j * 256 + threadIdx.x] =
            b[j * blockDim.x * gridDim.x + i] + c[i * 4 + j];
        }

        __syncthreads();
        d[i + 8]      = a_s[threadIdx.x];
        e[i * 8]      = bc_s[threadIdx.x * 4];
    }
    ```
   For each of the following memory accesses, specify whether they are **coalesced**, **uncoalesced**, or **coalescing is not applicable**:
   - a. The access to array `a` of line 05, **Answer**: Coalesced
   - b. The access to array `a_s` of line 05, **Answer**: Not applicable, because it is shared memory, here accessing the values is not slow
   - c. The access to array `b` of line 07, **Answer**: Coalesced, threads in the same block are contiguous
   - d. The access to array `c` of line 07, **Answer**: Uncoalesced
   - e. The access to array `bc_s` of line 07, **Answer**: Not applicable
   - f. The access to array `a_s` of line 10, **Answer**: Not applicable
   - g. The access to array `d` of line 10, **Answer**: Coalesced
   - h. The access to array `bc_s` of line 11, **Answer**: Not applicable
   - i. The access to array `e` of line 11, **Answer**: Uncoalesced 

4. What is the floating-point to global-memory access ratio (in OP/B) of each of the following matrix–matrix multiplication kernels?
   - a. The simple kernel described in Chapter 3, *Multidimensional Grids and Data*, without any optimizations applied. **Answer**: Here each thread does a work of $\frac{2 \text{ops}}{2 * 4\text{bytes}} = 0.25 \text{op/B}$
   - b. The kernel described in Chapter 5, *Memory Architecture and Data Locality*, with shared-memory tiling applied using a tile size of 32 × 32. **Answer**: $\frac{2 * 32 * \frac{k}{32}}{2 * 4 * \frac{k}{32}} = \frac{2 * 32 \text{ops}}{2 * 4 \text{bytes}} = \frac{64}{8} = 8 \text{op/b}$
   - c. The kernel described in this chapter with shared-memory tiling applied using a tile size of 32 × 32 and thread coarsening applied using a coarsening factor of 4. **Answer**: Coarsening is only done in the M dimension of matrix B so we have $\frac{(2 * 32 * 4) * \frac{K}{32}}{(1 * 4 + 4 * 4) * \frac{k}{32}} = \frac{256 \text{ops}}{20 \text{bytes}} = 12.8 \text{op/b}$ 

*Here we are only counting that data is saved in row-major and we don't count the saving of data into the C matrix output*