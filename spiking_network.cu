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

__global__ void ApproximateSpikeGradient(float* spike_gradient_approximation,
                                         const float* current_membrane_voltages,
                                         const float threshold_voltage, const float gradient_scaling_factor,
                                         const float num_batches, const float num_neurons) {

    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    // thread specific ID
    int tID = b * num_neurons + n;

    if (tID < num_batches * num_neurons) {
        spike_gradient_approximation[tID] = gradient_scaling_factor *
                fmaxf(0.0f, 1 - fabsf(current_membrane_voltages[tID] - threshold_voltage) / threshold_voltage);
    }
}

__global__ void CalculateTotalGradient(float* resulting_total_dE_dv,
                                       const float* current_partial_dE_dv, const float* current_sum_over_k,
                                       const int num_batches, const int num_neurons) {

    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {

        resulting_total_dE_dv[tID] = current_partial_dE_dv[tID] + current_sum_over_k[tID];
    }

}

__global__ void CalculateVoltageGradient(float* dv_k_dv_j,
                                         const float* spike_gradient_approximation, const float* recurrent_weights,
                                         const float decay_factor, const float threshold_voltage,
                                         const int num_batches, const int num_neurons) {

    const int batch_ID = blockIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.z * blockDim.z + threadIdx.z;

    const int tID = batch_ID * num_neurons * num_neurons + k * num_neurons + j;

    if (tID < num_batches * num_neurons * num_neurons) {

        if (k == j) {
            // possible restucture: load W_rec at k, j into shared memory across batches,
            // this would result in only one load from global memory and num_batches loads from shared memory
            dv_k_dv_j[tID] = decay_factor + (recurrent_weights[k * num_neurons + j] - threshold_voltage) *
                    spike_gradient_approximation[batch_ID * num_neurons + j];
        }
        else {
            dv_k_dv_j[tID] = recurrent_weights[k * num_neurons + j] *
                    spike_gradient_approximation[batch_ID * num_neurons + j];
        }
    }
}

__global__ void SumUpComponent(float* output_matrix,
                               const float* component,
                               const int size_first_dim, const int size_second_dim) {

    const int first_dim_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int second_dim_id = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    const int tID = first_dim_id * size_second_dim + second_dim_id;

    if (tID < size_first_dim * size_second_dim) {

        output_matrix[tID] += component[tID];
    }
}

__global__ void CheckIfNanInput(bool *is_it_nan,
                                const float* data,
                                const int time_step, const int first_dim_size, const int second_dim_size) {

    const int first_dim_ID = blockIdx.x * blockDim.x + threadIdx.x;
    const int second_dim_ID = blockIdx.y * blockDim.y + threadIdx.y;
    const int tID = first_dim_ID * second_dim_size + second_dim_ID;

    if (tID < first_dim_size * second_dim_size) {
        if (isnan(data[tID])) {
            *is_it_nan = true;
        }

        __syncthreads();

        if ((*is_it_nan) & (tID == 0)) {
            printf("Timestep %d: Input weights are nan\n", time_step);
            *is_it_nan = false;
        }
    }

}

__global__ void CheckIfNanRecurrent(bool *is_it_nan,
                                    const float* data,
                                    const int time_step, const int first_dim_size, const int second_dim_size) {

    const int first_dim_ID = blockIdx.x * blockDim.x + threadIdx.x;
    const int second_dim_ID = blockIdx.y * blockDim.y + threadIdx.y;
    const int tID = first_dim_ID * second_dim_size + second_dim_ID;

    if (tID < first_dim_size * second_dim_size) {
        if (isnan(data[tID])) {
            *is_it_nan = true;
        }

        __syncthreads();

        if ((*is_it_nan) & (tID == 0)) {
            printf("Timestep %d: Recurrent weights are nan\n", time_step);
            *is_it_nan = false;
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

    // printf("[INFO] Calculating batch matmul\n");

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

    // printf("[INFO] Batch matmul completed\n");

    // printf("[INFO] Running time steps\n");

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

   // printf("[INFO] Time steps completed\n");
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
                              float* current_spike_gradient, float* current_dv_k_dv_j, float* current_sum_over_k,
                              float* current_partial_dE_dv, float* previous_total_dE_dv, float* current_total_dE_dv,
                              float* dE_dW_in_component, float* dE_dW_rec_component,
                              const float* time_series_data, const float* resulting_voltages, const float* resulting_activations,
                              const float* partial_dE_dv,
                              const float* W_rec,
                              bool* input_nan, bool* recurrent_nan) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream_v2(cublas_handle, device.stream());

    // ceiling division of the threads into the number of needed blocks
    const int maximum_threads_per_block = 1024; // hard limit set by hardware
    const int max_threads_per_dimension_of_block = 32; // sqrt of maximum_threads_per_block

    // ceiling division by max_threads_per_dimension_of_block
    const int num_neuron_blocks = (num_neurons + max_threads_per_dimension_of_block - 1) / max_threads_per_dimension_of_block;
    const int num_batch_blocks = (num_batches + max_threads_per_dimension_of_block - 1) / max_threads_per_dimension_of_block;
    const int num_input_blocks = (num_input_channels + max_threads_per_dimension_of_block - 1) / max_threads_per_dimension_of_block;

    dim3 regularKernelGridSize(num_batch_blocks, num_neuron_blocks, 1);
    dim3 inputWeightsGridSize = dim3(num_neuron_blocks, num_input_blocks, 1);
    dim3 recurrentWeightsGridSize = dim3(num_neuron_blocks, num_neuron_blocks, 1);
    dim3 voltageGradientGridSize(num_batches, num_neuron_blocks, num_neuron_blocks);

    dim3 regularKernelBlockSize(max_threads_per_dimension_of_block, max_threads_per_dimension_of_block, 1);
    dim3 voltageGradientBlockSize(1, max_threads_per_dimension_of_block, max_threads_per_dimension_of_block);

    SetToValue<<<inputWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_in,
                                                                                     0.0,
                                                                                     num_neurons, num_input_channels);

    SetToValue<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_rec,
                                                                                         0.0,
                                                                                         num_neurons, num_neurons);

    // First time step t = num_time_steps - 1
    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_input_data,
                                                                                         time_series_data, num_time_steps - 1,
                                                                                         num_batches, num_input_channels);

    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_neuron_activations,
                                                                                         resulting_activations, num_time_steps - 1,
                                                                                         num_batches, num_neurons);

    // At the last time step (t=num_timesteps - 1), the partial gradient is equal to the total gradient
    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_total_dE_dv,
                                                                                         partial_dE_dv, num_time_steps - 1,
                                                                                         num_batches, num_neurons);

    // use the total derivative to calculate the derivative wrt the different weights

    // INPUT WEIGHTS

    cublasSgemm_v2(cublas_handle,
                   CUBLAS_OP_N, CUBLAS_OP_T,
                   num_input_channels, num_neurons, num_batches,
                   &alpha,
                   current_input_data, num_input_channels,
                   current_total_dE_dv, num_neurons,
                   &beta,
                   dE_dW_in_component, num_input_channels);

    SumUpComponent<<<inputWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_in,
                                                                                         dE_dW_in_component,
                                                                                         num_neurons, num_input_channels);
    // RECURRENT WEIGHTS

    cublasSgemm_v2(cublas_handle,
                   CUBLAS_OP_N, CUBLAS_OP_T,
                   num_neurons, num_neurons, num_batches,
                   &alpha,
                   current_neuron_activations, num_neurons,
                   current_total_dE_dv, num_neurons,
                   &beta,
                   dE_dW_rec_component, num_neurons);

    SumUpComponent<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_rec,
                                                                                             dE_dW_rec_component,
                                                                                             num_neurons, num_neurons);

    // Remaining time steps
    for (int t = num_time_steps - 2; t >= 0; --t) {

        CheckIfNanInput<<<inputWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(input_nan,
                                                                                              dE_dW_in,
                                                                                              t, num_neurons, num_input_channels);

        CheckIfNanRecurrent<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(recurrent_nan,
                                                                                                      dE_dW_rec,
                                                                                                      t, num_neurons, num_neurons);

        // Select data
        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_input_data,
                                                                                             time_series_data, t,
                                                                                             num_batches, num_input_channels);

        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                                             resulting_voltages, t,
                                                                                             num_batches, num_neurons);

        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_neuron_activations,
                                                                                             resulting_activations, t,
                                                                                             num_batches, num_neurons);

        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_partial_dE_dv,
                                                                                             partial_dE_dv, t,
                                                                                             num_batches, num_neurons);

        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(previous_total_dE_dv,
                                                                                             current_total_dE_dv, 0,
                                                                                             num_batches, num_neurons);

        // Calculate total derivative of loss function wrt the membrane voltages for this time step
        ApproximateSpikeGradient<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_spike_gradient,
                                                                                                        current_membrane_voltages,
                                                                                                        threshold_voltage, gradient_scaling_factor,
                                                                                                        num_batches, num_neurons);

        CalculateVoltageGradient<<<voltageGradientGridSize, voltageGradientBlockSize, 0, device.stream()>>>(current_dv_k_dv_j,
                                                                                                            current_spike_gradient, W_rec,
                                                                                                            decay_factor, threshold_voltage,
                                                                                                            num_batches, num_neurons);

        // sum over index k across all batches
        cublasSgemmStridedBatched(cublas_handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  num_neurons, num_neurons, num_neurons,
                                  &alpha,
                                  current_dv_k_dv_j, num_neurons, num_neurons * num_neurons,
                                  previous_total_dE_dv, num_neurons, num_neurons,
                                  &beta,
                                  current_sum_over_k, num_neurons, num_neurons,
                                  num_batches);

        // Addition of the partial derivative to the recurrent component
        CalculateTotalGradient<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_total_dE_dv,
                                                                                                      current_partial_dE_dv,
                                                                                                      current_sum_over_k,
                                                                                                      num_batches, num_neurons);
        // use the total derivative to calculate the derivative wrt the different weights

        // INPUT WEIGHTS

        cublasSgemm_v2(cublas_handle,
                       CUBLAS_OP_N, CUBLAS_OP_T,
                       num_input_channels, num_neurons, num_batches,
                       &alpha,
                       current_input_data, num_input_channels,
                       current_total_dE_dv, num_neurons,
                       &beta,
                       dE_dW_in_component, num_input_channels);

        SumUpComponent<<<inputWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_in,
                                                                                             dE_dW_in_component,
                                                                                             num_neurons, num_input_channels);
        // RECURRENT WEIGHTS

        cublasSgemm_v2(cublas_handle,
                   CUBLAS_OP_N, CUBLAS_OP_T,
                   num_neurons, num_neurons, num_batches,
                   &alpha,
                   current_neuron_activations, num_neurons,
                   current_total_dE_dv, num_neurons,
                   &beta,
                   dE_dW_rec_component, num_neurons);

        SumUpComponent<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_rec,
                                                                                                 dE_dW_rec_component,
                                                                                                 num_neurons, num_neurons);
    }
    /*
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

    // dot product of z(t) total_dE_dv(t) for each time step
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
    SumUpComponents<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_in_components,
                                                                                          dE_dW_in,
                                                                                          num_time_steps,
                                                                                          num_input_channels,
                                                                                          num_neurons);

    SumUpComponents<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_rec_components,
                                                                                          dE_dW_rec,
                                                                                          num_time_steps,
                                                                                          num_neurons,
                                                                                          num_neurons);
                                                                                          */
}
#endif // GOOGLE_CUDA