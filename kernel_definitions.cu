#include <stdio.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "cuda_profiler_api.h"
#include "kernel_definitions.h"

__host__ __device__ bool readBinArray(const uint8_t *array, const int index) {
    return ((uint8_t)array[index/8] >> (index % 8)) & 1UL;
}

__device__ void warpReduce(volatile float *shared_data, int tID, int blockSize) {
    if (blockSize >= 64) { shared_data[tID] += shared_data[tID + 32]; }
    if (blockSize >= 32) { shared_data[tID] += shared_data[tID + 16]; }
    if (blockSize >= 16) { shared_data[tID] += shared_data[tID + 8]; }
    if (blockSize >= 8) { shared_data[tID] += shared_data[tID + 4]; }
    if (blockSize >= 4) { shared_data[tID] += shared_data[tID + 2]; }
    if (blockSize >= 2) { shared_data[tID] += shared_data[tID + 1]; }
}

__global__ void float_inner_kernel(const int N, const int K, const int M, const float *W, const float *X, float *Z) {
    extern __shared__ float shared_data[];


    int tID = threadIdx.x;
    int weightsID =  blockIdx.x * K + tID;
    int dataID = M * tID + blockIdx.y;

    int outputID_x = blockIdx.x;
    int outputID_y = blockIdx.y;

    /* SegFault guards:
     * dataID: K*M long, access element i and i + K*M/2
     * guard: K*M - K*M/2 = K*M/2
     *
     *  weightsID: N*K long, access element j and j+K/2
     *  guard: N*K - K/2
     */
    if ((dataID < K * M/2) && (weightsID < (N * K - K/2))) {
        shared_data[tID] = W[weightsID] * X[dataID] + W[weightsID + int(K/2)] * X[dataID + int(M*K/2)];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("%f\n", shared_data[tID]);
    }

    if (blockDim.x >= 1024) { if (tID < 512) { shared_data[tID] += shared_data[tID + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tID < 256) { shared_data[tID] += shared_data[tID + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tID < 128) { shared_data[tID] += shared_data[tID + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tID < 64) { shared_data[tID] += shared_data[tID + 64]; } __syncthreads(); }

    /*
    for (unsigned int i = blockDim.x/2; i > 32; i >>= 1) {
        if (tID < i) {
            shared_data[tID] += shared_data[tID + i];
        }
        __syncthreads();
    }
    */

    if (threadIdx.x == 0) {
        printf("%f\n", shared_data[tID]);
    }

    if (tID < 32) { warpReduce(shared_data, tID, blockDim.x); }


    if (threadIdx.x == 0) {
        printf("%f\n", shared_data[tID]);
    }

    if (tID == 0) {

        Z[M * outputID_x + outputID_y] = shared_data[0];
    }

}

void cuda_float_inner(const int N, const int K, const int M, const float *W, const float *X, float *Z) {
    float *dev_X, *dev_Z, *dev_W;

    cudaMalloc((void **)&dev_X, K * M * sizeof(float));
    cudaMalloc((void **)&dev_W, N * K * sizeof(float));
    cudaMalloc((void **)&dev_Z, N * M * sizeof(float));

    // copy data from host to device
    cudaMemcpy(dev_X, X, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W, W, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // specify device parameters
    dim3 blockSize = dim3(K/2, 1, 1);
    dim3 gridSize = dim3(N, M, 1);
    size_t sharedMemSize = K/2 * sizeof(float);


    // launch kernel
    float_inner_kernel<<<gridSize, blockSize, sharedMemSize>>>(N, K, M, dev_W, dev_X, dev_Z);

    // copy data back to device
    cudaMemcpy(Z, dev_Z, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // free Device memory
    cudaFree(dev_X);
    cudaFree(dev_W);
    cudaFree(dev_Z);
}

__global__ void bool_inner_kernel(const int N, const int K, const int M, const float *W, const uint8_t *X, float *Z) {
    extern __shared__ float shared_data[];

    int tID = threadIdx.x;
    int weightsID =  blockIdx.x * K + tID;
    int dataID = M * tID + blockIdx.y;

    int outputID_x = blockIdx.x;
    int outputID_y = blockIdx.y;

    /* SegFault guards:
     * dataID: K*M long, access element i and i + K*M/2
     * guard: K*M - K*M/2 = K*M/2
     *
     *  weightsID: N*K long, access element j and j+K/2
     *  guard: N*K - K/2
     */
    if ((dataID < K * M/2) && (weightsID < (N * K - K/2))) {
        shared_data[tID] = W[weightsID] * readBinArray(X, dataID) +
                W[weightsID + int(K/2)] * readBinArray(X, dataID + int(M*K/2));
    }

    __syncthreads();

    if (blockDim.x >= 1024) { if (tID < 512) { shared_data[tID] += shared_data[tID + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tID < 256) { shared_data[tID] += shared_data[tID + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tID < 128) { shared_data[tID] += shared_data[tID + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tID < 64) { shared_data[tID] += shared_data[tID + 64]; } __syncthreads(); }


    /*
    for (unsigned int i = blockDim.x/2; i > 32; i >>= 1) {
        if (tID < i) {
            shared_data[tID] += shared_data[tID + i];
        }
        __syncthreads();
    }
    */

    if (tID < 32) { warpReduce(shared_data, tID, blockDim.x); }

    if (tID == 0) {

        Z[M * outputID_x + outputID_y] = shared_data[0];
    }
}

void cuda_bool_inner(const int N, const int K, const int M, const float *W, const uint8_t *X, float *Z) {
    float *dev_Z, *dev_W;
    uint8_t *dev_X;

    cudaMalloc((void **)&dev_X, (K * M + 7)/8 * sizeof(uint8_t));
    cudaMalloc((void **)&dev_W, N * K * sizeof(float));
    cudaMalloc((void **)&dev_Z, N * M * sizeof(float));

    // copy data from host to device
    cudaMemcpy(dev_X, X, (K * M + 7)/8 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W, W, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // specify device parameters
    dim3 blockSize = dim3(K/2, 1, 1);
    dim3 gridSize = dim3(N, M, 1);
    size_t sharedMemSize = K/2 * sizeof(float);


    // launch kernel
    bool_inner_kernel<<<gridSize, blockSize, sharedMemSize>>>(N, K, M, dev_W, dev_X, dev_Z);

    // copy data back to device
    cudaMemcpy(Z, dev_Z, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // free Device memory
    cudaFree(dev_X);
    cudaFree(dev_W);
    cudaFree(dev_Z);
}

int main() {
    const int N = 8;
    const int K = 4;
    const int M = 2;

    float W[N * K], X[K * M], Z[N * M], Z_binary[N * M];

    uint8_t X_binary[(K * M +7)/8];
    memset(X_binary, 0, (K * M +7)/8 * sizeof(uint8_t));

    read_csv_to_array<float>(W, N, K, "./data/inputs/W_test.csv");
    read_csv_to_array<float>(X, K, M, "./data/inputs/X_test.csv");

    for (int i = 0; i < K * M; ++i) {
        X_binary[i/8] |= ((unsigned int)X[i] << (i % 8));
    }

    cuda_float_inner(N, K, M, W, X, Z);
    cuda_bool_inner(N, K, M, W, X_binary, Z_binary);

    result_to_csv<float>(Z, N, M, "./data/outputs/calculated_result.csv");
    result_to_csv<float>(Z_binary, N, M, "./data/outputs/calculated_result_binary.csv");


    return 0;
}






