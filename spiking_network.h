#include <unsupported/Eigen/CXX11/Tensor>
#include "cublas_v2.h"

using GPUDevice = Eigen::GpuDevice;

struct ForwardPass {
public:
    ForwardPass(cublasHandle_t cublas_handle,
                int num_batches,
                int num_neurons,
                int num_input_channels,
                int num_time_steps,
                float decay_factor,
                float threshold_voltage);

    void operator()(::tensorflow::OpKernelContext* ctx, const GPUDevice &device,
                    const float *W_in, const float *W_rec,
                    const float *time_series_data,
                    float *base_activity,
                    float *current_v, float *current_z, float *current_base_activity,
                    float *resulting_voltages, float *resulting_activities);

private:
    cublasHandle_t cublas_handle;

    int num_batches;
    int num_neurons;
    int num_input_channels;
    int num_time_steps;

    float threshold_voltage;
    float decay_factor;
};

struct BackwardPass {
public:
    BackwardPass(cublasHandle_t cublas_handle,
                 int num_batches,
                 int num_neurons,
                 int num_input_channels,
                 int num_timesteps,
                 float decay_factor,
                 float threshold_voltage,
                 float gradient_scaling_factor);

    void operator()(::tensorflow::OpKernelContext* ctx, const GPUDevice &device,
                    float* dE_dW_in, float* dE_dW_rec,
                    float* current_input_data, float* current_membrane_voltages, float* current_neuron_activations,
                    float* current_partial_dE_dv, float* previous_total_dE_dv, float* current_total_dE_dv,
                    float* total_dE_dv,
                    float* dE_dW_in_components, float* dE_dW_rec_components,
                    const float* time_series_data, const float* resulting_voltages, const float* resulting_activations,
                    const float* partial_dE_dv,
                    const float* W_rec);

private:
    cublasHandle_t cublas_handle;

    int num_batches;
    int num_neurons;
    int num_input_channels;
    int num_time_steps;

    float threshold_voltage;
    float decay_factor;
    float gradient_scaling_factor;
};