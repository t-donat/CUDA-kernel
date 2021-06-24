#include <unsupported/Eigen/CXX11/Tensor>

using GPUDevice = Eigen::GpuDevice;

struct ForwardPass {
public:
    ForwardPass(int num_neurons,
                             int num_input_channels,
                             int num_timesteps,
                             float decay_factor,
                             float threshold_voltage);

    void operator()(::tensorflow::OpKernelContext* ctx, const GPUDevice &device,
                    const float *W_in, const float *W_rec, float *v, float *z,
                    const float *timeseries_data, float *base_activity, float *current_input,
                    float *resulting_voltages, float *resulting_activities);

private:
    int num_neurons;
    int num_input_channels;
    int num_timesteps;

    float threshold_voltage;
    float decay_factor;
};