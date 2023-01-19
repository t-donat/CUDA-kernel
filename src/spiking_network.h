#include <unsupported/Eigen/CXX11/Tensor>
#include "cublas_v2.h"

using GPUDevice = Eigen::GpuDevice;

struct ForwardPass {
public:
    ForwardPass(cublasHandle_t cublas_handle,
                int batch_size,
                int num_neurons,
                int num_input_channels,
                int num_time_steps,
                float threshold_voltage,
                float delta_t);

    void operator()(::tensorflow::OpKernelContext* ctx, const GPUDevice &device,
                    const float *W_in, const float *W_rec, const float *membrane_time_constants,
                    const float *time_series_data, float *base_activity,
                    float *membrane_decay_factors,
                    float *current_membrane_voltages, float *current_neuron_activations,
                    float *current_input_component, float *current_neuron_component,
                    float *resulting_voltages, float *resulting_activations);

private:
    cublasHandle_t cublas_handle;

    int batch_size;
    int num_neurons;
    int num_input_channels;
    int num_time_steps;

    float threshold_voltage;
    float delta_t;
};

struct BackwardPass {
public:
    BackwardPass(cublasHandle_t cublas_handle,
                 int batch_size,
                 int num_neurons,
                 int num_input_channels,
                 int num_time_steps,
                 float threshold_voltage,
                 float delta_t,
                 float gradient_scaling_factor);

    void operator()(::tensorflow::OpKernelContext* ctx, const GPUDevice &device,
                    float* dE_dW_in, float* dE_dW_rec, float* dE_dmembrane_time_constants,
                    float* dE_dmembrane_decay_factors,
                    float* current_input_data, float* current_membrane_voltages, float* current_neuron_activations,
                    float* next_membrane_voltages, float* next_neuron_activations,
                    float* current_spike_gradient, float* current_partial_dE_dv, float* previous_total_dE_dv, float* current_total_dE_dv,
                    float* dE_dW_in_component, float* dE_dW_rec_component,
                    float* membrane_decay_factors,
                    const float* time_series_data, const float* resulting_voltages, const float* resulting_activations,
                    const float* partial_dE_dv,
                    const float* W_rec, const float* membrane_time_constants);

private:
    cublasHandle_t cublas_handle;

    int batch_size;
    int num_neurons;
    int num_input_channels;
    int num_time_steps;

    float threshold_voltage;
    float delta_t;
    float gradient_scaling_factor;
};