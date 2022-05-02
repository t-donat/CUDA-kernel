#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <unsupported/Eigen/CXX11/Tensor>

#include "spiking_network.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "cublas_v2.h"

#include "general_kernel_definitions.cu"
#include "forward_pass_kernel_definitions.cu"
#include "backward_pass_kernel_definitions.cu"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

ForwardPass::ForwardPass(cublasHandle_t cublas_handle,
                         int batch_size,
                         int num_neurons,
                         int num_input_channels,
                         int num_time_steps,
                         float threshold_voltage,
                         float delta_t) :
                            cublas_handle(cublas_handle),
                            batch_size(batch_size),
                            num_neurons(num_neurons),
                            num_input_channels(num_input_channels),
                            num_time_steps(num_time_steps),
                            threshold_voltage(threshold_voltage),
                            delta_t(delta_t)
                         { };

void ForwardPass::operator()(OpKernelContext* ctx, const GPUDevice &device,
                             const float *W_in, const float *W_rec, const float *membrane_time_constants,
                             const float *time_series_data, float *base_activity,
                             float* membrane_decay_factors,
                             float *current_membrane_voltages, float *current_neuron_activations,
                             float *current_input_component, float *current_neuron_component,
                             float *resulting_voltages, float *resulting_activations)
{
    static float alpha = 1.0f;
    static float beta = 0.0f;
    //static cudaError_t exit_status;

    // ceiling division of the threads into the number of needed blocks
    int maximum_threads_per_block = 1024; // hard limit set by hardware
    int max_threads_per_dimension_of_block = 32; // sqrt of maximum_threads_per_block

    // ceiling division by max_threads_per_dimension_of_block
    // (x + y - 1) / y = 1 + (x - 1) / y
    // avoids overflows from x + y
    const int num_neuron_blocks = 1 + (num_neurons - 1) / max_threads_per_dimension_of_block;
    const int num_batch_blocks = 1 + (batch_size - 1) / max_threads_per_dimension_of_block;
    const int num_membrane_decay_blocks = 1 + (num_neurons - 1) / maximum_threads_per_block;

    // printf("%d\n", num_batch_blocks);

    dim3 kernelGridSize(num_batch_blocks, num_neuron_blocks, 1);
    dim3 membraneTimeConstantGridSize(num_membrane_decay_blocks, 1, 1);

    dim3 kernelBlockSize(max_threads_per_dimension_of_block, max_threads_per_dimension_of_block, 1);

    // set values of v and z to guarentee that they are initialized to 0
    SetToValue<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_membrane_voltages, 0.0f,
                                                                        batch_size, num_neurons);

    SetToValue<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_neuron_activations, 0.0f,
                                                                        batch_size, num_neurons);

    CalculateMembraneDecayFactors<<<membraneTimeConstantGridSize, kernelBlockSize, 0, device.stream()>>>(membrane_decay_factors,
                                                                                                membrane_time_constants,
                                                                                                delta_t, num_neurons);

    cublasSetStream_v2(cublas_handle, device.stream());

    // printf("[INFO] Calculating batch matmul\n");

    // initial large GEMM of input time series data with input weights
    cublasSgemmStridedBatched(cublas_handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              num_neurons, batch_size, num_input_channels,
                              &alpha,
                              W_in, num_input_channels, 0,
                              time_series_data, num_input_channels, num_input_channels * batch_size,
                              &beta,
                              base_activity, num_neurons, batch_size * num_neurons,
                              num_time_steps);

    // printf("[INFO] Batch matmul completed\n");

    // printf("[INFO] Running time steps\n");

    // iterate though the time series
    for (int t = 0; t < num_time_steps; t++) {

        CopyFromInput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_input_component,
                                                                               base_activity, t,
                                                                               batch_size, num_neurons);

        cublasSgemm_v2(cublas_handle,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       num_neurons, batch_size, num_neurons,
                       &alpha,
                       W_rec, num_neurons,
                       current_neuron_activations, num_neurons,
                       &beta,
                       current_neuron_component, num_neurons);

        MembraneVoltageUpdateRule<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                                           current_input_component, current_neuron_component,
                                                                                           membrane_decay_factors,
                                                                                           batch_size, num_neurons);

        NeuronActivationUpdateRule<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(current_neuron_activations,
                                                                                            current_membrane_voltages, threshold_voltage,
                                                                                            batch_size, num_neurons);

        CopyToOutput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(resulting_voltages,
                                                                              current_membrane_voltages, t,
                                                                              batch_size, num_neurons);

        CopyToOutput<<<kernelGridSize, kernelBlockSize, 0, device.stream()>>>(resulting_activations,
                                                                              current_neuron_activations, t,
                                                                              batch_size, num_neurons);
    }

   // printf("[INFO] Time steps completed\n");
}

BackwardPass::BackwardPass(cublasHandle_t cublas_handle,
                           int batch_size,
                           int num_neurons,
                           int num_input_channels,
                           int num_time_steps,
                           float threshold_voltage,
                           float delta_t,
                           float gradient_scaling_factor) :
                              cublas_handle(cublas_handle),
                              batch_size(batch_size),
                              num_neurons(num_neurons),
                              num_input_channels(num_input_channels),
                              num_time_steps(num_time_steps),
                              threshold_voltage(threshold_voltage),
                              delta_t(delta_t),
                              gradient_scaling_factor(gradient_scaling_factor)
                           { };

void BackwardPass::operator()(OpKernelContext* ctx, const GPUDevice &device,
                              float* dE_dW_in, float* dE_dW_rec, float* dE_dmembrane_time_constants,
                              float* dE_dmembrane_decay_factors,
                              float* current_input_data, float* current_membrane_voltages, float* current_neuron_activations, float* next_membrane_voltages,
                              float* current_spike_gradient, float* current_partial_dE_dv, float* previous_total_dE_dv, float* current_total_dE_dv,
                              float* dE_dW_in_component, float* dE_dW_rec_component,
                              float* membrane_decay_factors,
                              const float* time_series_data, const float* resulting_voltages, const float* resulting_activations,
                              const float* partial_dE_dv,
                              const float* W_rec, const float* membrane_time_constants) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream_v2(cublas_handle, device.stream());

    // ceiling division of the threads into the number of needed blocks
    const int maximum_threads_per_block = 1024; // hard limit set by hardware
    const int max_threads_per_dimension_of_block = 32; // sqrt of maximum_threads_per_block

    // ceiling division by max_threads_per_dimension_of_block
    // (x + y - 1) / y = 1 + (x - 1) / y
    // avoids overflows from x + y
    const int num_neuron_blocks = 1 + (num_neurons - 1) / max_threads_per_dimension_of_block;
    const int num_batch_blocks = 1 + (batch_size - 1) / max_threads_per_dimension_of_block;
    const int num_input_blocks = 1 + (num_input_channels - 1) / max_threads_per_dimension_of_block;
    const int num_membrane_decay_blocks = 1 + (num_neurons - 1) / maximum_threads_per_block;

    dim3 regularKernelGridSize(num_batch_blocks, num_neuron_blocks, 1);
    dim3 inputDataGridSize(num_batch_blocks, num_input_blocks, 1);
    dim3 inputWeightsGridSize(num_neuron_blocks, num_input_blocks, 1);
    dim3 recurrentWeightsGridSize(num_neuron_blocks, num_neuron_blocks, 1);
    dim3 membraneTimeConstantGridSize(num_membrane_decay_blocks, 1, 1);
    //dim3 voltageGradientGridSize(batch_size, num_neuron_blocks, num_neuron_blocks);

    dim3 regularKernelBlockSize(max_threads_per_dimension_of_block, max_threads_per_dimension_of_block, 1);
    // dim3 voltageGradientBlockSize(1, max_threads_per_dimension_of_block, max_threads_per_dimension_of_block);

    SetToValue<<<inputWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_in,
                                                                                     0.0,
                                                                                     num_neurons, num_input_channels);

    SetToValue<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_rec,
                                                                                         0.0,
                                                                                         num_neurons, num_neurons);

    CalculateMembraneDecayFactors<<<membraneTimeConstantGridSize, regularKernelBlockSize, 0, device.stream()>>>(membrane_decay_factors,
                                                                                                                membrane_time_constants,
                                                                                                                delta_t, num_neurons);

    SetMembraneTimeConstantDerivativeToZero<<<membraneTimeConstantGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dmembrane_decay_factors, num_neurons);
    //SetMembraneTimeConstantDerivativeToZero<<<membraneTimeConstantGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dmembrane_time_constants, num_neurons);

    // First time step t = num_time_steps - 1
    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_input_data,
                                                                                         time_series_data, num_time_steps - 1,
                                                                                         batch_size, num_input_channels);

    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                                         resulting_voltages, num_time_steps - 1,
                                                                                         batch_size, num_input_channels);

    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(next_membrane_voltages,
                                                                                         resulting_voltages, num_time_steps - 2,
                                                                                         batch_size, num_neurons);

    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_neuron_activations,
                                                                                         resulting_activations, num_time_steps - 1,
                                                                                         batch_size, num_neurons);

    // At the last time step (t=num_timesteps - 1), the partial gradient is equal to the total gradient
    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_total_dE_dv,
                                                                                         partial_dE_dv, num_time_steps - 1,
                                                                                         batch_size, num_neurons);

    // use the total derivative to calculate the derivative wrt the different weights

    // INPUT WEIGHTS

    cublasSgemm_v2(cublas_handle,
                   CUBLAS_OP_N, CUBLAS_OP_T,
                   num_input_channels, num_neurons, batch_size,
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
                   num_neurons, num_neurons, batch_size,
                   &alpha,
                   current_neuron_activations, num_neurons,
                   current_total_dE_dv, num_neurons,
                   &beta,
                   dE_dW_rec_component, num_neurons);

    SumUpComponent<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_rec,
                                                                                             dE_dW_rec_component,
                                                                                             num_neurons, num_neurons);

    // MEMBRANE TIME CONSTANTS

    CalculateDecayFactorDerivative<<<membraneTimeConstantGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dmembrane_decay_factors,
                                                                                                                 current_total_dE_dv, next_membrane_voltages,
                                                                                                                 batch_size, num_neurons);


    /*
    CalculateDecayFactorDerivative<<<membraneTimeConstantGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dmembrane_time_constants,
                                                                                                                 current_total_dE_dv, next_membrane_voltages,
                                                                                                                 batch_size, num_neurons);
    */



    /*
    // FOR TESTING

    CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(membrane_derivative_components_tensor,
                                                                                        membrane_tc_derivative_component, num_time_steps - 1,
                                                                                        1, num_neurons);

    CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(membrane_derivative_progress,
                                                                                        dE_dmembrane_time_constants, num_time_steps-2,
                                                                                        1, num_neurons);

    CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(total_gradients,
                                                                                        current_total_dE_dv, num_time_steps - 1,
                                                                                        batch_size, num_neurons);

    CopyToOutput<<<inputWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(input_gradients,
                                                                                       dE_dW_in_component, num_time_steps - 1,
                                                                                       num_neurons, num_input_channels);

    CopyToOutput<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(recurrent_gradients,
                                                                                           dE_dW_rec_component, num_time_steps - 1,
                                                                                           num_neurons, num_neurons);

     CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(next_voltage_tensor,
                                                                                        next_membrane_voltages, num_time_steps - 1,
                                                                                        batch_size, num_neurons);

    // FOR TESTING

    */


    // Remaining time steps
    for (int t = num_time_steps - 2; t >= 1; --t) {

        // Select data
        CopyFromInput<<<inputDataGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_input_data,
                                                                                         time_series_data, t,
                                                                                         batch_size, num_input_channels);

        /*
        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                                             resulting_voltages, t,
                                                                                             batch_size, num_neurons);
        */

        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                                             next_membrane_voltages, 0,
                                                                                             batch_size, num_neurons);


        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(next_membrane_voltages,
                                                                                             resulting_voltages, t-1,
                                                                                             batch_size, num_neurons);

        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_neuron_activations,
                                                                                             resulting_activations, t,
                                                                                             batch_size, num_neurons);

        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_partial_dE_dv,
                                                                                             partial_dE_dv, t,
                                                                                             batch_size, num_neurons);

        CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(previous_total_dE_dv,
                                                                                             current_total_dE_dv, 0,
                                                                                             batch_size, num_neurons);

        // Calculate total derivative of loss function wrt the membrane voltages for this time step
        ApproximateSpikeGradient<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_spike_gradient,
                                                                                                        current_membrane_voltages,
                                                                                                        threshold_voltage, gradient_scaling_factor,
                                                                                                        batch_size, num_neurons);

        CalculateTotalGradient<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_total_dE_dv,
                                                                                                      current_partial_dE_dv, previous_total_dE_dv,
                                                                                                      current_spike_gradient,
                                                                                                      W_rec, membrane_decay_factors,
                                                                                                      threshold_voltage,
                                                                                                      batch_size, num_neurons);

        /*
         * OLD CODE FOR CALCULATING THE TOTAL GRADIENT
         *
         * from backward pass function signature
         * float* current_dv_k_dv_j, float* current_sum_over_k,
        CalculateVoltageGradient<<<voltageGradientGridSize, voltageGradientBlockSize, 0, device.stream()>>>(current_dv_k_dv_j,
                                                                                                            current_spike_gradient, W_rec,
                                                                                                            decay_factor, threshold_voltage,
                                                                                                            batch_size, num_neurons);

        // sum over index k across all batches
        cublasSgemmStridedBatched(cublas_handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  num_neurons, num_neurons, num_neurons,
                                  &alpha,
                                  current_dv_k_dv_j, num_neurons, num_neurons * num_neurons,
                                  previous_total_dE_dv, num_neurons, num_neurons,
                                  &beta,
                                  current_sum_over_k, num_neurons, num_neurons,
                                  batch_size);

        // Addition of the partial derivative to the recurrent component
        CalculateTotalGradient<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_total_dE_dv,
                                                                                                      current_partial_dE_dv,
                                                                                                      current_sum_over_k,
                                                                                                      batch_size, num_neurons);
        */

        // use the total derivative to calculate the derivative wrt the different weights

        // INPUT WEIGHTS

        cublasSgemm_v2(cublas_handle,
                       CUBLAS_OP_N, CUBLAS_OP_T,
                       num_input_channels, num_neurons, batch_size,
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
                   num_neurons, num_neurons, batch_size,
                   &alpha,
                   current_neuron_activations, num_neurons,
                   current_total_dE_dv, num_neurons,
                   &beta,
                   dE_dW_rec_component, num_neurons);

        SumUpComponent<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_rec,
                                                                                                 dE_dW_rec_component,
                                                                                                 num_neurons, num_neurons);

        // MEMBRANE TIME CONSTANTS

        CalculateDecayFactorDerivative<<<membraneTimeConstantGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dmembrane_decay_factors,
                                                                                                                     current_total_dE_dv, next_membrane_voltages,
                                                                                                                     batch_size, num_neurons);

        /*

        CalculateDecayFactorDerivative<<<membraneTimeConstantGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dmembrane_time_constants,
                                                                                                                     current_total_dE_dv, next_membrane_voltages,
                                                                                                                     batch_size, num_neurons);

        */


        /*

        // FOR TESTING

        CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(membrane_derivative_components_tensor,
                                                                                        membrane_tc_derivative_component, t,
                                                                                        1, num_neurons);

        CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(membrane_derivative_progress,
                                                                                        dE_dmembrane_time_constants, t-1,
                                                                                        1, num_neurons);

        CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(total_gradients,
                                                                                            current_total_dE_dv, t,
                                                                                            batch_size, num_neurons);

        CopyToOutput<<<inputWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(input_gradients,
                                                                                           dE_dW_in_component, t,
                                                                                           num_neurons, num_input_channels);

        CopyToOutput<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(recurrent_gradients,
                                                                                               dE_dW_rec_component, t,
                                                                                               num_neurons, num_neurons);

        CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(next_voltage_tensor,
                                                                                        next_membrane_voltages, t,
                                                                                        batch_size, num_neurons);

        // FOR TESTING
        */

    }

    // Last BPTT time step,
    // Select data
    CopyFromInput<<<inputDataGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_input_data,
                                                                                         time_series_data, 0,
                                                                                         batch_size, num_input_channels);


    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                                         next_membrane_voltages, 0,
                                                                                         batch_size, num_neurons);

    /*
    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_membrane_voltages,
                                                                                         resulting_voltages, 0,
                                                                                         batch_size, num_neurons);
*/
    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_neuron_activations,
                                                                                             resulting_activations, 0,
                                                                                             batch_size, num_neurons);

    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_partial_dE_dv,
                                                                                             partial_dE_dv, 0,
                                                                                             batch_size, num_neurons);

    CopyFromInput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(previous_total_dE_dv,
                                                                                             current_total_dE_dv, 0,
                                                                                             batch_size, num_neurons);

    // Calculate total derivative of loss function wrt the membrane voltages for this time step
    ApproximateSpikeGradient<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_spike_gradient,
                                                                                                        current_membrane_voltages,
                                                                                                        threshold_voltage, gradient_scaling_factor,
                                                                                                        batch_size, num_neurons);

    CalculateTotalGradient<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(current_total_dE_dv,
                                                                                                      current_partial_dE_dv, previous_total_dE_dv,
                                                                                                      current_spike_gradient,
                                                                                                      W_rec, membrane_decay_factors,
                                                                                                      threshold_voltage,
                                                                                                      batch_size, num_neurons);



    // use the total derivative to calculate the derivative wrt the different weights

    // INPUT WEIGHTS

    cublasSgemm_v2(cublas_handle,
                       CUBLAS_OP_N, CUBLAS_OP_T,
                       num_input_channels, num_neurons, batch_size,
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
                   num_neurons, num_neurons, batch_size,
                   &alpha,
                   current_neuron_activations, num_neurons,
                   current_total_dE_dv, num_neurons,
                   &beta,
                   dE_dW_rec_component, num_neurons);

    SumUpComponent<<<recurrentWeightsGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dW_rec,
                                                                                                 dE_dW_rec_component,
                                                                                                 num_neurons, num_neurons);

    // MEMBRANE TIME CONSTANTS


    CalculateMembraneTimeConstantDerivative<<<membraneTimeConstantGridSize, regularKernelBlockSize, 0, device.stream()>>>(dE_dmembrane_time_constants,
                                                                                                                          dE_dmembrane_decay_factors,
                                                                                                                          membrane_decay_factors, membrane_time_constants,
                                                                                                                          delta_t, num_neurons);

    /*
    CopyToOutput<<<regularKernelGridSize, regularKernelBlockSize, 0, device.stream()>>>(next_voltage_tensor,
                                                                                        next_membrane_voltages, 0,
                                                                                        batch_size, num_neurons);
    */


}
#endif // GOOGLE_CUDA