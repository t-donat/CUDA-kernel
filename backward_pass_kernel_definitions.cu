
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
        spike_gradient_approximation[tID] = gradient_scaling_factor * fmaxf(0.0f, 1 - fabsf(current_membrane_voltages[tID] - threshold_voltage) / threshold_voltage);
    }
}

__global__ void CalculateTotalGradient(float* current_total_dE_dv,
                                       const float* current_partial_dE_dv, const float* previous_total_dE_dv,
                                       const float* current_spike_gradient, const float* recurrent_weights,
                                       const float* membrane_decay_factors, const float threshold_voltage,
                                       const int num_batches, const int num_neurons) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {

        float dE_dv_k, dv_k_dv_j;

        float result = current_partial_dE_dv[tID];

        for (int k = 0; k < num_neurons; ++k) {

            dE_dv_k = previous_total_dE_dv[b * num_neurons + k];

            if (k == j) {
                dv_k_dv_j = membrane_decay_factors[j] + recurrent_weights[k * num_neurons + j] * current_spike_gradient[tID];
            }

            else {
                dv_k_dv_j = recurrent_weights[k * num_neurons + j] * current_spike_gradient[tID];
            }

            result += dE_dv_k * dv_k_dv_j;
        }

        current_total_dE_dv[tID] = result;
    }
}

__global__ void CalculateDecayFactorDerivative(float* dE_dmembrane_decay_factors_component,
                                               const float* current_total_dE_dv, const float* next_membrane_voltages,
                                               const int num_batches, const int num_neurons) {

    const int maximum_threads_per_block = 1024;
    const int neuron_Id = blockIdx.x * maximum_threads_per_block + threadIdx.y * blockDim.x + threadIdx.x;

    float result = 0.0f;
    int access_Id;

    if (neuron_Id < num_neurons) {
        for (int b = 0; b < num_batches; ++b) {
            access_Id = b * num_neurons + neuron_Id;
            result += current_total_dE_dv[access_Id] * next_membrane_voltages[access_Id];
        }
        dE_dmembrane_decay_factors_component[neuron_Id] += result;
    }
}



__global__ void CalculateMembraneTimeConstantDerivative(float* dE_dmembrane_time_constants,
                                                        const float* dE_dmembrane_decay_factors,
                                                        const float* membrane_decay_factors, const float* membrane_time_constants,
                                                        const float delta_t,
                                                        const int num_neurons) {

    const int maximum_threads_per_block = 1024;
    const int neuron_Id = blockIdx.x * maximum_threads_per_block + threadIdx.y * blockDim.x + threadIdx.x;

    if (neuron_Id < num_neurons) {
        // Saving into local memory makes the squaring by multiplication more efficient by avoiding loading twice from global memory
        float current_decay_factor = membrane_decay_factors[neuron_Id];
        float current_decay_factor_derivative = dE_dmembrane_decay_factors[neuron_Id];
        float current_membrane_time_constant = membrane_time_constants[neuron_Id];

        dE_dmembrane_time_constants[neuron_Id] = current_decay_factor_derivative * delta_t / (current_membrane_time_constant * current_membrane_time_constant) * current_decay_factor;
    }
}

/*
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
*/
