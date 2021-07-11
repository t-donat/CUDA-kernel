#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <unsupported/Eigen/CXX11/Tensor>


#include "spiking_network.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "cublas_v2.h"

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

__global__ void NeuronActivationUpdateRule(const float* v_membrane, float* z, float v_th) {
    int tID = blockDim.y * threadIdx.x + threadIdx.y;

    if (tID < blockDim.x * blockDim.y ) {
        if (v_membrane[tID] < v_th) {
            z[tID] = 0.0f;
        }
        else {
            z[tID] = 1.0f;
        }
    }
}


__global__ void MembraneVoltageUpdateRule(float* v, const float* z, const float* input, const float v_th) {
    int tID = blockDim.y * threadIdx.x + threadIdx.y;

    if (tID < blockDim.x * blockDim.y) {
        v[tID] += input[tID] - v_th * z[tID];
    }
}

__global__ void CopyFromInput(const float* base_voltage_activity, const int timestep, float* out) {
   int tID = blockDim.y * threadIdx.x + threadIdx.y;

    if (tID < blockDim.x * blockDim.y ) {
        out[tID] = base_voltage_activity[blockDim.x * blockDim.y * timestep + tID];
    }
}

__global__ void CopyToOutput(const float* in, const int timestep, float *result_array) {
    int tID = blockDim.y * threadIdx.x + threadIdx.y;

    if (tID < blockDim.x * blockDim.y ) {
        result_array[blockDim.x * blockDim.y * timestep + tID] = in[tID];
    }
}

__global__ void SetToValue(float *in, const float value) {
    int tID = blockDim.y * threadIdx.x + threadIdx.y;

    if (tID < blockDim.x * blockDim.y) {
        in[tID] = value;
    }
}

ForwardPass::ForwardPass(cublasHandle_t cublas_handle,
                         int batch_size,
                         int num_neurons,
                         int num_input_channels,
                         int num_timesteps,
                         float decay_factor,
                         float threshold_voltage) :
                            cublas_handle(cublas_handle),
                            batch_size(batch_size),
                            num_neurons(num_neurons),
                            num_input_channels(num_input_channels),
                            num_timesteps(num_timesteps),
                            decay_factor(decay_factor),
                            threshold_voltage(threshold_voltage)
                         { };

void ForwardPass::operator()(OpKernelContext* ctx, const GPUDevice &device,
                             const float *W_in, const float *W_rec, float *v, float *z,
                             const float *timeseries_data, float *base_activity, float *current_input,
                             float *resulting_voltages, float *resulting_activities)
{
    static float alpha = 1.0f;
    static float beta = 0.0f;
    //static cudaError_t exit_status;

    dim3 kernelBlockSize(batch_size, num_neurons, 1);
    dim3 kernelGridSize(1, 1, 1);

    // set values of v and z to guarentee that they are initialized to 0
    SetToValue<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(v, 0.0f);
    SetToValue<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(z, 0.0f);

    cublasSetStream(cublas_handle, device.stream());

    // initial large GEMM of input time series data with input weights
    cublasSgemmStridedBatched(cublas_handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              num_neurons, batch_size, num_input_channels,
                              &alpha,
                              W_in, num_neurons, 0,
                              timeseries_data, num_input_channels, batch_size * num_input_channels,
                              &beta,
                              base_activity, num_neurons, batch_size * num_neurons,
                              num_timesteps);


    // iterate though the time series
    for (int t = 0; t < num_timesteps; t++) {
        CopyFromInput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(base_activity, t, current_input);

        cublasSgemm_v2(cublas_handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       num_neurons, batch_size, num_neurons,
                       &alpha,
                       W_rec, num_neurons,
                       z, num_neurons,
                       &decay_factor,
                       v, num_neurons);

        MembraneVoltageUpdateRule<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(v, z, current_input, threshold_voltage);
        NeuronActivationUpdateRule<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(v, z, threshold_voltage);

        CopyToOutput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(v, t, resulting_voltages);
        CopyToOutput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(z, t, resulting_activities);
    }
}

#endif // GOOGLE_CUDA