#include <stdio.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

template <typename T>
void result_to_csv(T *result, const int N, const int M, const char *filename) {
    std::ofstream ResultFile;
    ResultFile.open(filename);

    for (int k = 0; k < N; k++) {
        for (int l = 0; l < M; l++) {

            ResultFile << result[k * M + l];

            if (l < M-1) {
                ResultFile << ",";
            }
            else {
                ResultFile << "\n";
            }
        }
    }

    ResultFile.close();
/*
    for (int i = 0; i < N * M; ++i) {
        std::cout << result[i];
        if (i%M == M-1) {
            std::cout << std::endl;
        }

        else {
            std::cout << ", ";
        }
    }
*/
}

template<typename T>
void read_csv_to_array(T *result_array, const int N, const int M, const char *filename) {
    std::ifstream target_file;
    target_file.open(filename);

    char temp_for_delimiter;
    std::string file_line;

    if (target_file.good()) {
        for (int j = 0; j < N; ++j) {
            std::getline(target_file, file_line);
            std::stringstream iss(file_line);
            iss >> result_array[j * M];

            for (int i = 1; i < M; ++i) {
                iss >> temp_for_delimiter >> result_array[j * M + i];
            }
        }
    }

    target_file.close();

}


__global__ void float_inner_kernel(const int N, const int K, const int M, const float *W, const float *X, float *Z) {
    extern __shared__ float shared_data[];

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

void cuda_float_inner(const int N, const int K, const int M, const float *W, const float *X, float *Z) {
    float *dev_X, *dev_Z, *dev_W;

    cudaMalloc((void **)&dev_X, K * M * sizeof(float));
    cudaMalloc((void **)&dev_W, N * K * sizeof(float));
    cudaMalloc((void **)&dev_Z, N * M * sizeof(float));

    // copy data from host to device
    cudaMemcpy(dev_X, X, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W, W, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // specify device parameters
    dim3 blockSize = dim3(K, 1, 1);
    dim3 gridSize = dim3(N, M, 1);
    size_t sharedMemSize = K * sizeof(float);

    // launch kernel
    float_inner_kernel<<<gridSize, blockSize, sharedMemSize>>>(N, K, M, dev_W, dev_X, dev_Z);

    // copy data back to device
    cudaMemcpy(Z, dev_Z, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // free Device memory
    cudaFree(dev_X);
    cudaFree(dev_W);
    cudaFree(dev_Z);
}

int main() {
    int N = 64;
    int K = 32;
    int M = 64;

    float W[N * K], X[K * M], Z[N * M];

    read_csv_to_array<float>(W, N, K, "./data/inputs/W.csv");
    read_csv_to_array<float>(X, K, M, "./data/inputs/X.csv");

    cuda_float_inner(N, K, M, W, X, Z);

    result_to_csv<float>(Z, N, M, "./data/outputs/calculated_result.csv");

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