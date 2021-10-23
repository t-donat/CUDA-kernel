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

__global__ void NeuronActivationUpdateRule(float* neuron_activations,
                                           const float* membrane_voltages, const float v_th,
                                           const int num_batches, const int num_neurons) {
    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {
        if (membrane_voltages[tID] < v_th) {
            neuron_activations[tID] = 0.0f;
        }
        else {
            neuron_activations[tID] = 1.0f;
        }
    }
}


__global__ void MembraneVoltageUpdateRule(float* intermediate_membrane_voltages,
                                          const float* neuron_activations, const float* base_activity, const float v_th,
                                          const int num_batches, const int num_neurons) {
    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {
        intermediate_membrane_voltages[tID] += base_activity[tID] - v_th * neuron_activations[tID];
    }
}

__global__ void SetToValue(float* input_matrix,
                           const float set_value,
                           const int num_batches, const int num_neurons) {
    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {
        input_matrix[tID] = set_value;
    }
}

__global__ void CopyFromInput(float* output_matrix,
                              const float* input_tensor, const int time_step,
                              const int num_batches, const int num_neurons) {
    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {
        output_matrix[tID] = input_tensor[num_batches * num_neurons * time_step + tID];
    }
}


__global__ void CopyToOutput(float *result_tensor,
                             const float* input_matrix, const int time_step,
                             const int num_batches, const int num_neurons) {

    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {
        result_tensor[num_batches * num_neurons * time_step + tID] = input_matrix[tID];
    }
}

__global__ void CalculateTotalGradient(float* resulting_total_dE_dv,
                                       const float decay_factor, const float v_th, const float gradient_scaling_factor,
                                       const float* partial_dE_dv, const float* previous_total_dE_dv, const float* current_membrane_voltages,
                                       const int num_neurons, const int num_batches,
                                       const float* W_rec) {

    float dv_k_dv_j;
    float previous_dE_dv_k;
    //float current_spike_gradient;

    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {

        // partial derivative of loss function E wrt this neuron's membrane voltage
        float total_dE_dv = partial_dE_dv[tID];


        float current_spike_gradient = gradient_scaling_factor *
                fmaxf(0.0f, 1 - fabsf(current_membrane_voltages[tID] - v_th) / v_th);

        for (int k = 0; k < num_neurons; ++k) {

            if (k == j) {
                dv_k_dv_j = decay_factor + (W_rec[k * num_neurons + j] - v_th) * current_spike_gradient;
            }
            else {
                dv_k_dv_j = W_rec[k * num_neurons + j] * current_spike_gradient;
            }

            previous_dE_dv_k = previous_total_dE_dv[b * num_neurons + k];

            total_dE_dv += dv_k_dv_j * previous_dE_dv_k;
        }

        resulting_total_dE_dv[tID] = total_dE_dv;
    }

}

__global__ void SumUpComponents(float* output_matrix,
                                const float* component_tensor,
                                const int num_time_steps, const int size_first_dim, const int size_second_dim) {

    int first_dim_id = blockIdx.x * blockDim.x + threadIdx.x;
    int second_dim_id = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = first_dim_id * size_second_dim + second_dim_id;

    if (tID < size_first_dim * size_second_dim) {
        for (int t = 0; t < num_time_steps; ++t) {
            output_matrix[tID] += component_tensor[t * size_first_dim * size_second_dim + tID];
        }
    }
}

ForwardPass::ForwardPass(cublasHandle_t cublas_handle,
                         int num_batches,
                         int num_neurons,
                         int num_input_channels,
                         int num_time_steps,
                         float decay_factor,
                         float threshold_voltage) :
                            cublas_handle(cublas_handle),
                            num_batches(num_batches),
                            num_neurons(num_neurons),
                            num_input_channels(num_input_channels),
                            num_time_steps(num_time_steps),
                            decay_factor(decay_factor),
                            threshold_voltage(threshold_voltage)
                         { };

void ForwardPass::operator()(OpKernelContext* ctx, const GPUDevice &device,
                             const float *W_in, const float *W_rec,
                             const float *time_series_data,
                             float *base_activity,
                             float *current_membrane_voltages, float *current_neuron_activations, float *current_base_activity,
                             float *resulting_voltages, float *resulting_activations)
{
    static float alpha = 1.0f;
    static float beta = 0.0f;
    //static cudaError_t exit_status;

    // ceiling division of the threads into the number of needed blocks
    int maximum_threads_per_block = 1024; // hard limit set by hardware
    int max_threads_per_dimension_of_block = 32; // sqrt of maximum_threads_per_block

    // ceiling division by max_threads_per_dimension_of_block
    int num_neuron_blocks = (num_neurons + max_threads_per_dimension_of_block - 1) / max_threads_per_dimension_of_block;
    int num_batch_blocks = (num_batches + max_threads_per_dimension_of_block - 1) / max_threads_per_dimension_of_block;

    dim3 kernelGridSize(num_batch_blocks, num_neuron_blocks, 1);
    dim3 kernelBlockSize(max_threads_per_dimension_of_block, max_threads_per_dimension_of_block, 1);

    // set values of v and z to guarentee that they are initialized to 0
    SetToValue<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_membrane_voltages, 0.0f,
                                                                        num_batches, num_neurons);

    SetToValue<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_neuron_activations, 0.0f,
                                                                        num_batches, num_neurons);

    cublasSetStream(cublas_handle, device.stream());
    //cublasSetStream_v2(cublas_handle, device.stream());

    printf("[INFO] Calculating batch matmul\n");

    // initial large GEMM of input time series data with input weights
    cublasSgemmStridedBatched(cublas_handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              num_neurons, num_batches, num_input_channels,
                              &alpha,
                              W_in, num_input_channels, 0,
                              time_series_data, num_input_channels, num_input_channels * num_batches,
                              &beta,
                              base_activity, num_neurons, num_batches * num_neurons,
                              num_time_steps);

    printf("[INFO] Batch matmul completed\n");

    printf("[INFO] Running time steps\n");

    // iterate though the time series
    for (int t = 0; t < num_time_steps; t++) {
        CopyFromInput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_base_activity,
                                                                               base_activity, t,
                                                                               num_batches, num_neurons);

        cublasSgemm_v2(cublas_handle,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       num_neurons, num_batches, num_neurons,
                       &alpha,
                       W_rec, num_neurons,
                       current_neuron_activations, num_neurons,
                       &decay_factor,
                       current_membrane_voltages, num_neurons);

        MembraneVoltageUpdateRule<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                                           current_neuron_activations, current_base_activity, threshold_voltage,
                                                                                           num_batches, num_neurons);

        NeuronActivationUpdateRule<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_neuron_activations,
                                                                                            current_membrane_voltages, threshold_voltage,
                                                                                            num_batches, num_neurons);

        CopyToOutput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(resulting_voltages,
                                                                              current_membrane_voltages, t,
                                                                              num_batches, num_neurons);

        CopyToOutput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(resulting_activations,
                                                                              current_neuron_activations, t,
                                                                              num_batches, num_neurons);
    }

    printf("[INFO] Time steps completed\n");
}

BackwardPass::BackwardPass(cublasHandle_t cublas_handle,
                           int num_batches,
                           int num_neurons,
                           int num_input_channels,
                           int num_time_steps,
                           float decay_factor,
                           float threshold_voltage,
                           float gradient_scaling_factor) :
                              cublas_handle(cublas_handle),
                              num_batches(num_batches),
                              num_neurons(num_neurons),
                              num_input_channels(num_input_channels),
                              num_time_steps(num_time_steps),
                              decay_factor(decay_factor),
                              threshold_voltage(threshold_voltage),
                              gradient_scaling_factor(gradient_scaling_factor)
                           { };

void BackwardPass::operator()(OpKernelContext* ctx, const GPUDevice &device,
                              float* dE_dW_in, float* dE_dW_rec,
                              float* current_input_data, float* current_membrane_voltages, float* current_neuron_activations,
                              float* current_partial_dE_dv, float* previous_total_dE_dv, float* current_total_dE_dv,
                              float* total_dE_dv,
                              float* dE_dW_in_components, float* dE_dW_rec_components,
                              const float* time_series_data, const float* resulting_voltages, const float* resulting_activations,
                              const float* partial_dE_dv,
                              const float* W_rec) {

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSetStream_v2(cublas_handle, device.stream());

    // ceiling division of the threads into the number of needed blocks
    int maximum_threads_per_block = 1024; // hard limit set by hardware
    int max_threads_per_dimension_of_block = 32; // sqrt of maximum_threads_per_block

    // ceiling division by max_threads_per_dimension_of_block
    int num_neuron_blocks = (num_neurons + max_threads_per_dimension_of_block - 1) / max_threads_per_dimension_of_block;
    int num_batch_blocks = (num_batches + max_threads_per_dimension_of_block - 1) / max_threads_per_dimension_of_block;

    dim3 kernelGridSize(num_batch_blocks, num_neuron_blocks, 1);
    dim3 kernelBlockSize(max_threads_per_dimension_of_block, max_threads_per_dimension_of_block, 1);

    for (int t = num_time_steps - 1; t >= 0; --t) {
        // Select data
        CopyFromInput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_input_data,
                                                                               time_series_data, t,
                                                                               num_batches, num_input_channels);

        CopyFromInput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                               resulting_voltages, t,
                                                                               num_batches, num_neurons);

        CopyFromInput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_neuron_activations,
                                                                               resulting_activations, t,
                                                                               num_batches, num_neurons);

        CopyFromInput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_partial_dE_dv,
                                                                               partial_dE_dv, t,
                                                                               num_batches, num_neurons);

        // TODO: alternative implementation: last time step outside of for loop and the remaining calculations from
        // TODO: t = num_time_steps - 2 till t = 0 in the for loop to avoid condition check in each iteration

        if (t == num_time_steps - 1) {
            // for the last time step, there is no previous total derivative, so it is set to 0
            SetToValue<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(previous_total_dE_dv,
                                                                                0.0,
                                                                                num_batches,
                                                                                num_neurons);
        }

        else {
            CopyFromInput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(previous_total_dE_dv,
                                                                                   total_dE_dv, t + 1,
                                                                                   num_batches, num_neurons);
        }

        // Calculate total derivative of loss function wrt the membrane voltages for this time step
        CalculateTotalGradient<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(
                current_total_dE_dv,
                decay_factor, threshold_voltage, gradient_scaling_factor,
                current_partial_dE_dv, previous_total_dE_dv, current_membrane_voltages,
                num_neurons, num_batches,
                W_rec);

        // save result to intermediate tensor
        CopyToOutput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(total_dE_dv,
                                                                              current_total_dE_dv, t,
                                                                              num_batches, num_neurons);

        //previous_total_dE_dv = curent_total_dE_dv;
    }

    // dot product of x(t) total_dE_dv(t) for each time step
    cublasSgemmStridedBatched(cublas_handle,
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              num_neurons, num_input_channels, num_batches,
                              &alpha,
                              total_dE_dv, num_neurons, num_neurons * num_batches,
                              time_series_data, num_input_channels, num_input_channels * num_batches,
                              &beta,
                              dE_dW_in_components, num_neurons, num_input_channels * num_neurons,
                              num_time_steps);

    // dot product of x(t) total_dE_dv(t) for each time step
    cublasSgemmStridedBatched(cublas_handle,
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              num_neurons, num_neurons, num_batches,
                              &alpha,
                              total_dE_dv, num_neurons, num_neurons * num_batches,
                              resulting_activations, num_neurons, num_neurons * num_batches,
                              &beta,
                              dE_dW_rec_components, num_neurons, num_neurons * num_neurons,
                              num_time_steps);

    // Sum up the components to calculate dE_dW_in & dE_dW_rec
    SumUpComponents<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(dE_dW_in_components,
                                                                             dE_dW_in,
                                                                             num_time_steps,
                                                                             num_input_channels,
                                                                             num_neurons);

    SumUpComponents<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(dE_dW_rec_components,
                                                                             dE_dW_rec,
                                                                             num_time_steps,
                                                                             num_neurons,
                                                                             num_neurons);
}
#endif // GOOGLE_CUDA