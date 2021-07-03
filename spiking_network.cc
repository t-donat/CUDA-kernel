#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <stdio.h>
#include <iostream>

#include "spiking_network.h"

static cublasHandle_t cublas_handle;

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("ForwardPass")
.Input("weights_input: float")
.Input("weights_recurrent: float")
.Input("time_series_data: float")
.Attr("decay_factor: float")
.Attr("threshold_voltage: float")
.Output("resulting_voltages: float")
.Output("resulting_activations: float");

class ForwardPassOp : public OpKernel {
private:
    float threshold_voltage;
    float decay_factor;

public:
    explicit ForwardPassOp(OpKernelConstruction* context) : OpKernel(context) {
        cublasCreate(&cublas_handle);

        // get the threshold voltage and decay factor
        OP_REQUIRES_OK(context, context->GetAttr("threshold_voltage", &threshold_voltage));
        OP_REQUIRES_OK(context, context->GetAttr("decay_factor", &decay_factor));

    }

    ~ForwardPassOp() override { cublasDestroy(cublas_handle); }

    void Compute(OpKernelContext* context) override {

        std::cout << "[INFO] Allocating input memory" << std::endl;
        // allocate input tensors and get their contents
        const Tensor& W_in_tensor = context->input(0);
        const Tensor& W_rec_tensor = context->input(1);
        const Tensor& time_series_tensor = context->input(2);

        std::cout << "[INFO] Getting dimensions" << std::endl;
        // get the values for the dimensions
        int num_neurons = static_cast<int>(W_rec_tensor.shape().dim_size(1));
        int num_time_steps = static_cast<int>(time_series_tensor.shape().dim_size(0));
        int num_input_channels = static_cast<int>(time_series_tensor.shape().dim_size(1));

        std::cout << "[INFO] Allocation output memory" << std::endl;
        // allocate output tensors
        TensorShape output_shape({num_time_steps, num_neurons});
        Tensor* resulting_voltages_tensor = nullptr;
        Tensor* resulting_activities_tensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &resulting_voltages_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &resulting_activities_tensor));

        std::cout << "[INFO] Allocating temporary memory" << std::endl;
        // allocate temporary/intermediate tensors
        Tensor v_tensor, z_tensor, current_input_tensor, base_voltage_activity_tensor;

        // shape of v and z in each time step
        TensorShape vector_shape({num_neurons, 1});

        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &v_tensor));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &z_tensor));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &current_input_tensor));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape, &base_voltage_activity_tensor));

        std::cout << "[INFO] Initializing functor" << std::endl;
        ForwardPass forward(cublas_handle,
                            num_neurons, num_input_channels, num_time_steps,
                            decay_factor, threshold_voltage);

        // exposing the data of the tensors
        /*
        auto W_in = W_in_tensor.flat<float>();
        auto W_rec = W_rec_tensor.flat<float>();
        auto time_series = time_series_tensor.flat<float>();

        auto v = v_tensor.flat<float>();
        auto z = z_tensor.flat<float>();
        auto current_input = current_input_tensor.flat<float>();

        auto base_voltage_activity = base_voltage_activity_tensor.flat<float>();

        auto resulting_voltages = resulting_activities_tensor->flat<float>();
        auto resulting_activities = resulting_activities_tensor->flat<float>();
        */
        std::cout << "[INFO] Running the forward pass" << std::endl;
        /*
        forward(context, context->eigen_gpu_device(),
                W_in.data(), W_rec.data(),
                v.data(), z.data(),
                time_series.data(), base_voltage_activity.data(),
                current_input.data(),
                resulting_voltages.data(), resulting_activities.data());
        */
        forward(context, context->eigen_gpu_device(),
                W_in_tensor.flat<float>().data(), W_rec_tensor.flat<float>().data(),
                v_tensor.flat<float>().data(), z_tensor.flat<float>().data(),
                time_series_tensor.flat<float>().data(), base_voltage_activity_tensor.flat<float>().data(),
                current_input_tensor.flat<float>().data(),
                resulting_voltages_tensor->flat<float>().data(), resulting_activities_tensor->flat<float>().data());


        std::cout << "[INFO] Operation completed" << std::endl;
    }
};

// TODO: CPU kernel version?

//#ifdef GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("ForwardPass").Device(DEVICE_GPU), ForwardPassOp);

//#endif // GOOGLE_CUDA