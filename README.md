# CUDA-kernel

## `mat_mul_prototype.cu`

This is my first protype for matrix multiplication implemented in CUDA. 

This algorithm calculates the following matrix multiplications:

> Z = W * X

Essentially, each element of the resulting matrix is calculated in a separate Thread Block. The necessary data is loaded into an array in shared memory of each block (since it can be quickly loaded from there). While loading, the elementwise multiplication is performed. In a second step, the shared memory array is summed up (using parallel reduction).

The kernel is written as a template to accept generic data types instead of only ints.

### TODO:
* limit template data types to numeric types
* Implement further optimizations with regard to the parallel reduction
* Profile if boolean values for X decrease computation time
