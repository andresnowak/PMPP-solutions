# Exercise

1. **Calculate the `P[0]` value in Fig. 7.3.**

$0 \cdot 1 + 3 \cdot 0 + 8 \cdot 5 + 2 \cdot 3 + 5 \cdot = 51$

2. **Consider performing a 1D convolution on array**  
   `N = {4, 1, 3, 2, 3}`  
   **with filter**  
   `F = {2, 1, 4}`  
   What is the resulting output array?

   $\{8,21,13,20,7\}$

3. **What do you think the following 1D convolution filters are doing?**  
   a. `[0 1 0]`  No change
   b. `[0 0 1]`  Left shift
   c. `[1 0 0]`  Right shift
   d. `[2 1/2 0 1/2]`  derivative $\frac{f(x + h) - f(x - h)}{2h} = \frac{f(x) - f(x-h)}{h}$ 
   e. `[1/3 1/3 1/3]` Mean

4. **Consider performing a 1D convolution on an array of size** `N`  
   **with a filter of size** `M`  
   a. How many ghost cells are there in total?   $\frac{M-1}{2} \cdot 2 = M-1$ 
   b. How many multiplications are performed if ghost cells are treated as multiplications (by 0)?  $N \cdot M$
   c. How many multiplications are performed if ghost cells are *not* treated as multiplications? $N \cdot M - (2r + 1 - 1) = N \cdot M - 2r = N \cdot M - M + 1$