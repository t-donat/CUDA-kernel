
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

__global__ void MembraneVoltageUpdateRule(float* current_membrane_voltages,
                                          const float* current_input_component, const float *current_neuron_component,
                                          const float* membrane_decay_factors,
                                          const int num_batches, const int num_neurons) {
    // batch index
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // neuron index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // thread ID index used to access data from array
    int tID = b * num_neurons + j;

    if (tID < num_batches * num_neurons) {
        current_membrane_voltages[tID] = membrane_decay_factors[j] * current_membrane_voltages[tID] + current_input_component[tID] + current_neuron_component[tID];
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