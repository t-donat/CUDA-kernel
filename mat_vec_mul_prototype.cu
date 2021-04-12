#include "stdio.h"

//#define N 8
//#define M 4

__global__ void forward(const int N, const int M, const int *W, const int *X, int *Z) {
    extern __shared__ int shared_data[];

    int tID = threadIdx.x;
    int inputID =  blockIdx.x * 2 * blockDim.x+ threadIdx.x;
    int outputID = blockIdx.x;

    shared_data[tID] = W[inputID] * X[tID] + W[inputID + blockDim.x] * X[tID + blockDim.x];

    __syncthreads();

    for (unsigned int i = blockDim.x/2; i > 0; i >>= 1) {
        if (tID < i) {
            shared_data[tID] += shared_data[tID + i];
        }
        __syncthreads();
    }

    if (tID == 0) {

        Z[outputID] = shared_data[0];
    }

}

__device__ void warpReduce(volatile int* temp, int tID) {
    //Manually unrolling the last 6 for-loop iterations since they only happen on a single warp
    //This removes the need for syncthreads and frees up all other warps

    temp[tID] += temp[tID + 32];
    temp[tID] += temp[tID + 16];
    temp[tID] += temp[tID + 8];
    temp[tID] += temp[tID + 4];
    temp[tID] += temp[tID + 2];
    temp[tID] += temp[tID + 1];
}

int main() {
    int N = 8;
    int M = 4;

    int W[N * M] = {-16, 2, 12, -16,
                    -11, -3, -11, 13,
                    25, 17, -4, 20,
                    13, -9, -6, -3,
                    -10, -12, 19, 21,
                    8, -20, 2, -22,
                    -6, -23, -8, -10,
                    -13, 22, 22, -5};


    int *Z;//, *W;
    int *dev_X, *dev_Z, *dev_W;

    // allocate memory
    Z = (int *)malloc(N * sizeof(int));
    //W = (int *)malloc(N * M * sizeof(int));

    // flatten the matrix to avoid difficulty with copying 2D array to Device
    /*for (int i = 0; i < N * M; i++) {
        W[i] = 1;
    }*/

    int X[M];

    for (int i = 0; i < M; i++) {
            X[i] = 1;
    }

    //X[2] = 0;
    //X[0] = 0;

    cudaMalloc((void **)&dev_X, M * sizeof(int));
    cudaMalloc((void **)&dev_W, N * M * sizeof(int));
    cudaMalloc((void **)&dev_Z, N * sizeof(int));

    // copy data from host to device
    cudaMemcpy(dev_X, X, M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W, W, N * M * sizeof(int), cudaMemcpyHostToDevice);

    // specify device parameters
    dim3 blockSize = dim3((M+1)/2, 1, 1);
    dim3 gridSize = dim3(N, 1, 1);
    size_t sharedMemSize = M * sizeof(int);

    // launch kernel
    forward<<<gridSize, blockSize, sharedMemSize>>>(N, M, dev_W, dev_X, dev_Z);

    // copy data back to device
    cudaMemcpy(Z, dev_Z, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int k = 0; k < N; k++) {
        printf("Z[%d]: %d\n", k, Z[k]);
    }

    free(Z);
    cudaFree(dev_X);
    cudaFree(dev_W);
    cudaFree(dev_Z);

    return 0;
}