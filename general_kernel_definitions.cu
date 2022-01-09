
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

__global__ void SetMembraneTimeConstantDerivativeToZero(float* membrane_decay_factors, const int num_neurons) {

    const int maximum_threads_per_block = 1024;
    const int neuron_Id = blockIdx.x * maximum_threads_per_block + threadIdx.y * blockDim.x + threadIdx.x;

    if (neuron_Id < num_neurons) {
        membrane_decay_factors[neuron_Id] = 0.0f;
    }
}

__global__ void CalculateMembraneDecayFactors(float* membrane_decay_factors,
                                              const float* membrane_time_constants, const float delta_t,
                                              const int num_neurons) {

    const int maximum_threads_per_block = 1024;
    const int neuron_Id = blockIdx.x * maximum_threads_per_block + threadIdx.y * blockDim.x + threadIdx.x;

    if (neuron_Id < num_neurons) {
        const float current_time_constant = membrane_time_constants[neuron_Id];
        const float x = delta_t / current_time_constant;
        membrane_decay_factors[neuron_Id] = expf(- x );
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


/*
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

*/