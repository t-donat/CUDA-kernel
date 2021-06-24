#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <unsupported/Eigen/CXX11/Tensor>

#include "spiking_network.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("ForwardPass")
.Input("weights_input: float")
.Input("weights_recurrent: float")
.Input("time_series_data: float")
.Input("decay_factor: float")
.Input("threshold_voltage: float")
.Output("resulting_voltages: float")
.Output("resulting_activations: float");

class ForwardPassOp : public OpKernel {
public:
    explicit ForwardPassOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // allocate input tensors
        const Tensor& W_in = context->input(0);
        const Tensor& W_rec = context->input(1);
        const Tensor& time_series_data = context->input(2);

        const Tensor& decay_factor_tensor = context->input(3);
        const Tensor& threshold_voltage_tensor = context->input(4);

        float decay_factor = decay_factor_tensor.flat<float>()(0);
        float threshold_voltage = threshold_voltage_tensor.flat<float>()(0);

        int num_neurons = static_cast<int>(W_rec.shape().dim_size(1));
        int num_time_steps = static_cast<int>(time_series_data.shape().dim_size(0));
        int num_input_channels = static_cast<int>(time_series_data.shape().dim_size(1));

        // allocate output tensors
        TensorShape output_shape({num_time_steps, num_neurons});
        Tensor* resulting_voltages = nullptr;
        Tensor* resulting_activities = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &resulting_voltages));
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &resulting_activities));

        // allocate temporary/intermediate tensors
        // TODO ase_voltage_activity: allocate_persistent? allocate as Const class?
        Tensor v_tensor, z_tensor ,current_input_tensor ,base_voltage_activity_tensor;

        // intermediate column vector shape for v and z
        TensorShape vector_shape({num_neurons, 1});

        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &v_tensor));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &z_tensor));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, vector_shape, &current_input_tensor));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape, &base_voltage_activity_tensor));

        // initialize v & z to zeros

        auto v_flat = v_tensor.flat<float>();
        auto z_flat = z_tensor.flat<float>();

        for (int i = 0; i < num_neurons; i++) {
            v_flat(i) = 0.0f;
            z_flat(i) = 0.0f;
        }


        ForwardPass forward(num_neurons, num_input_channels, num_time_steps, decay_factor, threshold_voltage);

        forward(context, context->eigen_gpu_device(),
                W_in.flat<float>().data(), W_rec.flat<float>().data(),
                v_flat.data(), z_flat.data(),
                time_series_data.flat<float>().data(), base_voltage_activity_tensor.flat<float>().data(),
                current_input_tensor.flat<float>().data(),
                resulting_voltages->flat<float>().data(), resulting_activities->flat<float>().data());



    }
};

// TODO: CPU kernel version?

//#ifdef GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("ForwardPass").Device(DEVICE_GPU), ForwardPassOp);

//#endif // GOOGLE_CUDA