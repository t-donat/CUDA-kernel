#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <stdio.h>
#include <iostream>

#include "spiking_network.h"

static cublasHandle_t forward_cublas_handle;
static cublasHandle_t backward_cublas_handle;

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("ForwardPass")
.Input("input_weights: float")
.Input("recurrent_weights: float")
.Input("time_series_data: float")
.Attr("decay_factor: float")
.Attr("threshold_voltage: float")
.Attr("debug_mode: bool = false")
.Output("resulting_voltages: float")
.Output("resulting_activations: float");

REGISTER_OP("BackwardPass")
.Input("voltages_partial_derivative: float")
.Input("weights_recurrent: float")
.Input("time_series_data: float")
.Input("resulting_voltages: float")
.Input("resulting_activations: float")
.Attr("decay_factor: float")
.Attr("threshold_voltage: float")
.Attr("gradient_scaling_factor: float")
.Attr("debug_mode: bool = false")
.Output("input_weights_derivative: float")
.Output("recurrent_weights_derivative: float");

class ForwardPassOp : public OpKernel {
private:
    float threshold_voltage;
    float decay_factor;
    bool debug_mode;

public:
    explicit ForwardPassOp(OpKernelConstruction* context) : OpKernel(context) {
        cublasCreate(&forward_cublas_handle);

        // get the threshold voltage and decay factor
        OP_REQUIRES_OK(context, context->GetAttr("threshold_voltage", &threshold_voltage));
        OP_REQUIRES_OK(context, context->GetAttr("decay_factor", &decay_factor));
        OP_REQUIRES_OK(context, context->GetAttr("debug_mode", &debug_mode));

    }

    ~ForwardPassOp() override { cublasDestroy(forward_cublas_handle); }

    void Compute(OpKernelContext* context) override {

        // Allocate input tensors and get their contents
        if (debug_mode) { std::cout << "[INFO] Allocating input memory" << std::endl; }

        const Tensor& input_weights = context->input(0); // (num_neurons x num_input_channels)
        const Tensor& recurrent_weights = context->input(1); // (num_neurons x num_neurons)
        const Tensor& time_series_data = context->input(2); // (num_time_steps x num_batches x num_input_channels)

        // Get the values for the dimensions
        if (debug_mode) { std::cout << "[INFO] Getting dimensions" << std::endl; }
        int num_neurons = static_cast<int>(recurrent_weights.shape().dim_size(0));
        int num_time_steps = static_cast<int>(time_series_data.shape().dim_size(0));
        int num_batches = static_cast<int>(time_series_data.shape().dim_size(1));
        int num_input_channels = static_cast<int>(time_series_data.shape().dim_size(2));

        // Allocate output tensors
        if (debug_mode) { std::cout << "[INFO] Allocation output memory" << std::endl; }

        TensorShape output_shape({num_time_steps, num_batches, num_neurons});
        Tensor* resulting_voltages = nullptr;
        Tensor* resulting_activations = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &resulting_voltages));
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &resulting_activations));

        // Allocate temporary/intermediate tensors
        if (debug_mode) { std::cout << "[INFO] Allocating temporary memory" << std::endl; }

        Tensor current_membrane_voltages, current_neuron_activations, current_base_activity;
        Tensor base_voltage_activity;

        // shape of v and z in each time step
        TensorShape vector_shape({num_batches, num_neurons});

        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &current_membrane_voltages));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &current_neuron_activations));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &current_base_activity));

        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape, &base_voltage_activity));


        if (debug_mode) { std::cout << "[INFO] Initializing functor" << std::endl; }

        /*
         *
        std::cout << "num_batches: " << num_batches << std::endl;
        std::cout << "num_neurons: " << num_neurons << std::endl;
        std::cout << "num_input_channels: " << num_input_channels << std::endl;
        std::cout << "num_time_steps: " << num_time_steps << std::endl;

        std::cout << "decay_factor: " << decay_factor << std::endl;
        std::cout << "threshold_voltage: " << threshold_voltage << std::endl;

        */

        ForwardPass forward(forward_cublas_handle,
                            num_batches,
                            num_neurons, num_input_channels, num_time_steps,
                            decay_factor, threshold_voltage);

        if (debug_mode) { std::cout << "[INFO] Running the forward pass" << std::endl; }

        forward(context, context->eigen_gpu_device(),
                input_weights.flat<float>().data(), recurrent_weights.flat<float>().data(),
                time_series_data.flat<float>().data(), base_voltage_activity.flat<float>().data(),
                current_membrane_voltages.flat<float>().data(), current_neuron_activations.flat<float>().data(), current_base_activity.flat<float>().data(),
                resulting_voltages->flat<float>().data(), resulting_activations->flat<float>().data());

        if (debug_mode) { std::cout << "[INFO] Forward pass completed" << std::endl; }
    }
};

class BackwardPassOp : public OpKernel {
private:
    float threshold_voltage;
    float decay_factor;
    float gradient_scaling_factor;
    bool debug_mode;

public:
    explicit BackwardPassOp(OpKernelConstruction* context) : OpKernel(context) {
        cublasCreate(&backward_cublas_handle);

        // get the threshold voltage and decay factor
        OP_REQUIRES_OK(context, context->GetAttr("threshold_voltage", &threshold_voltage));
        OP_REQUIRES_OK(context, context->GetAttr("decay_factor", &decay_factor));
        OP_REQUIRES_OK(context, context->GetAttr("gradient_scaling_factor", &gradient_scaling_factor));
        OP_REQUIRES_OK(context, context->GetAttr("debug_mode", &debug_mode));

    }

    ~BackwardPassOp() override { cublasDestroy(backward_cublas_handle); }

    void Compute(OpKernelContext* context) override {

        // Allocate input tensors and get their contents
        if (debug_mode) { std::cout << "[INFO] Allocating input memory" << std::endl; }

        const Tensor& partial_dE_dv_tensor = context->input(0); // (num_time_steps x num_batches x num_neurons)
        const Tensor& W_rec = context->input(1); // (num_neurons x num_neurons)
        const Tensor& time_series_data = context->input(2); // (num_time_steps x num_batches x num_input_channels)
        const Tensor& resulting_voltages= context->input(3); // (num_time_steps x num_batches x num_input_channels)
        const Tensor& resulting_activations = context->input(4); // (num_time_steps x num_batches x num_input_channels)

        // Get the values for the dimensions
        if (debug_mode) { std::cout << "[INFO] Getting dimensions" << std::endl; }

        int num_neurons = static_cast<int>(W_rec.shape().dim_size(0));
        int num_time_steps = static_cast<int>(time_series_data.shape().dim_size(0));
        int num_batches = static_cast<int>(time_series_data.shape().dim_size(1));
        int num_input_channels = static_cast<int>(time_series_data.shape().dim_size(2));

        // Allocate output tensors
        if (debug_mode) { std::cout << "[INFO] Allocating output memory" << std::endl; }

        TensorShape W_in_shape({num_neurons, num_input_channels});
        Tensor* dE_dW_in = nullptr;

        TensorShape W_rec_shape({num_neurons, num_neurons});
        Tensor* dE_dW_rec = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, W_in_shape, &dE_dW_in));
        OP_REQUIRES_OK(context, context->allocate_output(1, W_rec_shape, &dE_dW_rec));

        // Allocate temporary/intermediate tensors
        if (debug_mode) { std::cout << "[INFO] Allocating temporary memory" << std::endl; }

        Tensor current_input_data, current_membrane_voltages, current_neuron_activations;
        Tensor current_spike_gradient, current_partial_dE_dv, previous_total_dE_dv, current_total_dE_dv;
        //Tensor current_dv_k_dv_j, current_sum_over_k;
        Tensor dE_dW_in_components, dE_dW_rec_components;

        Tensor input_nan, recurrent_nan;

        // The default shape of most matrices in each time step
        TensorShape default_shape({num_batches, num_neurons});

        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({num_batches, num_input_channels}), &current_input_data));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, default_shape, &current_membrane_voltages));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, default_shape, &current_neuron_activations));

        //OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({num_batches, num_neurons, num_neurons}), &current_dv_k_dv_j));
        //OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, default_shape, &current_sum_over_k));

        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, default_shape, &current_spike_gradient));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, default_shape, &current_partial_dE_dv));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, default_shape, &previous_total_dE_dv));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, default_shape, &current_total_dE_dv));

        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({num_neurons, num_input_channels}), &dE_dW_in_components));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({num_neurons, num_neurons}), &dE_dW_rec_components));

        // Create backward pass functor
        if (debug_mode) { std::cout << "[INFO] Initializing functor" << std::endl; }

        BackwardPass backward(backward_cublas_handle,
                              num_batches,
                              num_neurons, num_input_channels, num_time_steps,
                              decay_factor, threshold_voltage, gradient_scaling_factor);


        if (debug_mode) { std::cout << "[INFO] Running the backward pass" << std::endl; }

        backward(context, context->eigen_gpu_device(),
                 dE_dW_in->flat<float>().data(), dE_dW_rec->flat<float>().data(),
                 current_input_data.flat<float>().data(), current_membrane_voltages.flat<float>().data(), current_neuron_activations.flat<float>().data(),
                 current_spike_gradient.flat<float>().data(),
                 current_partial_dE_dv.flat<float>().data(), previous_total_dE_dv.flat<float>().data(), current_total_dE_dv.flat<float>().data(),
                 dE_dW_in_components.flat<float>().data(), dE_dW_rec_components.flat<float>().data(),
                 time_series_data.flat<float>().data(), resulting_voltages.flat<float>().data(), resulting_activations.flat<float>().data(),
                 partial_dE_dv_tensor.flat<float>().data(),
                 W_rec.flat<float>().data());

        if (debug_mode) { std::cout << "[INFO] Backward pass completed" << std::endl; }
    }
};

// TODO: CPU kernel version?

//#ifdef GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("ForwardPass").Device(DEVICE_GPU), ForwardPassOp);
REGISTER_KERNEL_BUILDER(Name("BackwardPass").Device(DEVICE_GPU), BackwardPassOp);

//#endif // GOOGLE_CUDA