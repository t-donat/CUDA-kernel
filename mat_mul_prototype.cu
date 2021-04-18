#include <stdio.h>
#include "mat_mul_prototype.cuh"

#include "stdio.h"
#include <iostream>
#include <fstream>

template <typename T>
void result_to_csv(T *result, const int N, const int M) {
    std::ofstream ResultFile;
    ResultFile.open("./data/outputs/calculated_results.csv");
    
    for (int k = 0; k < N; k++) {
        for (int l = 0; l < M; l++) {

            ResultFile << result[k * M + l] << ",";
        }
        ResultFile << "\n";
    }

    ResultFile.close();
}

template<typename T>
__global__ void mat_mul_inner(const int N, const int K, const int M, const T *W, const T *X, T *Z) {
    extern __shared__ T shared_data[];

    int tID = threadIdx.x;
    int weightsID =  blockIdx.x * K + tID;
    int dataID = M * tID + blockIdx.y;

    int outputID_x = blockIdx.x;
    int outputID_y = blockIdx.y;

    if ((dataID < K * M) && (weightsID < N * K)) {
        shared_data[tID] = W[weightsID] * X[dataID];
    }

    __syncthreads();

    for (unsigned int i = blockDim.x/2; i > 0; i >>= 1) {
        if (tID < i) {
            shared_data[tID] += shared_data[tID + i];
        }
        __syncthreads();
    }

    if (tID == 0) {

        Z[M * outputID_x + outputID_y] = shared_data[0];
    }

}

template<typename T>
void cuda_mat_mul_inner(const int N, const int K, const int M, const T *W, const T *X, T *Z) {
    int *dev_X, *dev_Z, *dev_W;

    cudaMalloc((void **)&dev_X, K * M * sizeof(int));
    cudaMalloc((void **)&dev_W, N * K * sizeof(int));
    cudaMalloc((void **)&dev_Z, N * M * sizeof(int));

    // copy data from host to device
    cudaMemcpy(dev_X, X, K * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W, W, N * K * sizeof(int), cudaMemcpyHostToDevice);

    // specify device parameters
    dim3 blockSize = dim3(K, 1, 1);
    dim3 gridSize = dim3(N, M, 1);
    size_t sharedMemSize = K * sizeof(int);

    // launch kernel
    mat_mul_inner<int><<<gridSize, blockSize, sharedMemSize>>>(N, K, M, dev_W, dev_X, dev_Z);

    // copy data back to device
    cudaMemcpy(Z, dev_Z, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    // free Device memory
    cudaFree(dev_X);
    cudaFree(dev_W);
    cudaFree(dev_Z);
}

int main() {
    int N = 8;
    int K = 4;
    int M = 2;

    int X[K * M], Z[N * M];

    int W[N * K]  = {-16, 2, 12, -16,
                     -11, -3, -11, 13,
                     25, 17, -4, 20,
                     13, -9, -6, -3,
                     -10, -12, 19, 21,
                     8, -20, 2, -22,
                     -6, -23, -8, -10,
                     -13, 22, 22, -5};

    /*
    // Test Case: matrix of ints
    int W[N * K];
    for (int i = 0; i < N * K; i++) {
        W[i] = 1;
    }
    */

    for (int i = 0; i < K; i++) {
        X[M*i] = 1;
        X[M*i + 1] = 0;
    }

    cuda_mat_mul_inner<int>(N, K, M, W, X, Z);

    result_to_csv<int>(Z, N, M);

    return 0;
}





/*
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
*/