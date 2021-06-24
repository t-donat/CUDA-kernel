#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <unsupported/Eigen/CXX11/Tensor>


#include "spiking_network.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "cublas_v2.h"
cublasHandle_t cublasHandle = NULL;


using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;


__device__ void warpReduce(volatile float *shared_data, int tID, int blockSize) {
    if (blockSize >= 64) { shared_data[tID] += shared_data[tID + 32]; }
    if (blockSize >= 32) { shared_data[tID] += shared_data[tID + 16]; }
    if (blockSize >= 16) { shared_data[tID] += shared_data[tID + 8]; }
    if (blockSize >= 8) { shared_data[tID] += shared_data[tID + 4]; }
    if (blockSize >= 4) { shared_data[tID] += shared_data[tID + 2]; }
    if (blockSize >= 2) { shared_data[tID] += shared_data[tID + 1]; }
}

__global__ void FloatMatMulKernel(const int N, const int K, const int M, const float *W, const float *X, float *Z) {
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

    if (blockDim.x >= 1024) { if (tID < 512) { shared_data[tID] += shared_data[tID + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tID < 256) { shared_data[tID] += shared_data[tID + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tID < 128) { shared_data[tID] += shared_data[tID + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tID < 64) { shared_data[tID] += shared_data[tID + 64]; } __syncthreads(); }

    if (tID < 32) { warpReduce(shared_data, tID, blockDim.x); }

    if (tID == 0) {

        Z[M * outputID_x + outputID_y] = shared_data[0];
    }

}

__global__ void NeuronActivationUpdateRule(const float* v_membrane, const int N, float v_th, float* z) {
    int tID = threadIdx.x + blockDim.x * blockIdx.x;

    if (tID < N) {
        if (v_membrane[tID] < v_th) {
            z[tID] = 0.0f;
        }
        else {
            z[tID] = 1.0f;
        }
    }
}


__global__ void MembraneVoltageUpdateRule(float* v, const float* z, const float* input, const int N, const float v_th) {
    int tID = threadIdx.x + blockDim.x * blockIdx.x;

    if (tID < N) {
        v[tID] += input[tID] - v_th * z[tID];
    }
}

__global__ void CopyToInput(const float* base_voltage_activity, const int N, const int timestep, float* out) {
    int tID = threadIdx.x + blockDim.x * blockIdx.x;

    if (tID < N) {
        out[tID] = base_voltage_activity[N * timestep + tID];
    }
}

__global__ void CopyToOutput(const float* in, const int N, const int timestep, float *result_array) {
    int tID = threadIdx.x + blockDim.x * blockIdx.x;

    if (tID < N) {
        result_array[N * timestep + tID] = in[tID];
        //printf("%d, %d, %f\t", N * timestep + tID, tID,  result_array[N * timestep + tID]);
    }
}

void ForwardPass::operator()(OpKernelContext* ctx, const GPUDevice &device,
                             const float *W_in, const float *W_rec, float *v, float *z,
                             const float *timeseries_data, float *base_activity, float *current_input,
                             float *resulting_voltages, float *resulting_activities)
{
    static float alpha = 1.0f;
    static float beta = 0.0f;
    static cudaError_t exit_status;

    if (cublasHandle == NULL) {
    assert(cublasCreate_v2(&cublasHandle) == CUBLAS_STATUS_SUCCESS);
    assert(cublasSetStream_v2(cublasHandle, device.stream()) == CUBLAS_STATUS_SUCCESS);
    }
    /*
    static cublasHandle_t cublasHandle;
    cudaStream_t save_stream;

    cublasCreate_v2(&cublasHandle);
    cudaStreamCreate(&save_stream);
     cublasSetStream_v2(cublasHandle, save_stream);
    */
    // Actual Computation

    cublasSgemm_v2(cublasHandle,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   num_neurons, num_timesteps, num_input_channels,
                   &alpha,
                   W_in, num_neurons,
                   timeseries_data, num_input_channels,
                   &beta,
                   base_activity, num_neurons);

    dim3 kernelBlockSize(num_neurons, 1, 1);
    dim3 kernelGridSize(1, 1, 1);

    for (int t = 0; t < num_timesteps; t++) {
        CopyToInput<<<kernelBlockSize, kernelGridSize, 0, device.stream()>>>(base_activity, num_neurons, t, current_input);

        cublasSgemv_v2(cublasHandle,
                       CUBLAS_OP_T,
                       num_neurons, num_neurons,
                       &alpha,
                       W_rec, num_neurons,
                       z, 1,
                       &decay_factor,
                       v, 1);

        MembraneVoltageUpdateRule<<<kernelBlockSize, kernelGridSize, 0, device.stream()>>>(v, z, current_input, num_neurons, threshold_voltage);
        NeuronActivationUpdateRule<<<kernelBlockSize, kernelGridSize, 0, device.stream()>>>(v, num_neurons, threshold_voltage, z);

        CopyToOutput<<<kernelBlockSize, kernelGridSize, 0, device.stream()>>>(v, num_neurons, t, resulting_voltages);
        CopyToOutput<<<kernelBlockSize, kernelGridSize, 0, device.stream()>>>(z, num_neurons, t, resulting_activities);
    }
}

#endif // GOOGLE_CUDA