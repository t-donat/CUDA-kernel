import os
import time
import pickle
import numpy as np

import tensorflow as tf

from rsnn_utils.rsnn import initialize_weights, python_forward_pass, python_backward_pass, python_spike_gradient
from rsnn_utils.rsnn import calculate_spike_gradient
from rsnn_utils.data import find_indices_of_max_probabilities

target_directory = "../BCI_Data/Data/B_128"
# target_directory = "../BCI_Data/Data/dataset/dataset/B_128"
results_directory = "./Results"
num_repetitions = 10
neuron_numbers_to_test = [8, 16, 32] # , 64, 128, 256, 512, 1024]
debug_mode = False
save_intermediate_results = True

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

dt = 1/256
initial_membrane_time_constant = 100 / 1000

output_time_window = 100
threshold_voltage = 1

gradient_scaling_factor = 0.3


# -------------------------------------------------------------------
# GPU SETUP
# -------------------------------------------------------------------

physical_devices = tf.config.experimental.list_physical_devices('GPU')

for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# -------------------------------------------------------------------
# LOAD IN CUDA FUNCTIONS
# -------------------------------------------------------------------
shared_lib_path = os.path.join(os.path.dirname(__file__), "spiking_network.so")

if not os.path.exists(shared_lib_path):
    raise FileNotFoundError(f"Could not find shared library at expected path: {shared_lib_path}")
rsnn = tf.load_op_library(shared_lib_path)

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

with open(os.path.join(target_directory, "test_data_set.p"), "rb") as pickle_file:
    test_data_set = pickle.load(pickle_file)

samples = test_data_set[0][0]
labels = test_data_set[1][0]
del test_data_set

num_time_steps, batch_size, num_input_channels = samples.shape
_, num_output_channels = labels.shape

samples = samples.astype(np.float32)
samples_tf = tf.convert_to_tensor(samples, dtype=float)
labels_tf = tf.convert_to_tensor(labels, dtype=float)

# -------------------------------------------------------------------
# EXPERIMENT
# -------------------------------------------------------------------
results = {}

for num_neurons in neuron_numbers_to_test:
    print(f"Testing N={num_neurons}")
    forward_pass_python = []
    backward_pass_python = []

    forward_pass_cuda = []
    backward_pass_cuda = []

    (W_in, W_rec,
     W_out, tau_membrane) = initialize_weights(num_neurons, num_input_channels, num_output_channels,
                                               threshold_voltage, initial_membrane_time_constant)

    with tf.device("/cpu"):
        W_in_tf = tf.Variable(W_in, dtype=tf.float32)
        W_rec_tf = tf.Variable(W_rec, dtype=tf.float32)
        W_out_tf = tf.Variable(W_out, dtype=tf.float32)
        tau_membrane_tf = tf.Variable(tau_membrane, dtype=tf.float32)

    for _ in range(num_repetitions):
        # PYTHON
        start = time.time()
        python_voltages, python_activations = python_forward_pass(W_in, W_rec, tau_membrane,
                                                                  samples,
                                                                  threshold_voltage, dt)
        python_forward_duration = time.time() - start

        smoothed_spikes = np.stack([np.mean(python_activations[i-output_time_window:i], axis=0)
                                    if i >= output_time_window else np.zeros((batch_size, num_neurons))
                                    for i in range(num_time_steps)])

        network_output = np.dot(smoothed_spikes, W_out.T)

        softmax_output = np.exp(network_output - np.max(network_output))
        softmax_output = softmax_output / np.sum(softmax_output, axis=-1, keepdims=True)

        dE_dnetwork_output = np.zeros_like(network_output)

        for b in range(batch_size):
            current_data = softmax_output[:, b]
            argmax_index = np.argmax(current_data)

            argmax_time_step = argmax_index // num_neurons
            argmax_class = argmax_index % num_neurons

            predicted_distribution = current_data[argmax_time_step]
            ground_truth_distribution = labels[b]

            dE_dnetwork_output[argmax_time_step, b] = predicted_distribution - ground_truth_distribution

        dE_dsmoothed_spikes = np.dot(dE_dnetwork_output, W_out)

        dE_dz = np.stack([np.mean(dE_dsmoothed_spikes[i - output_time_window:i], axis=0)
                          if i >= output_time_window else np.zeros((batch_size, num_neurons))
                          for i in range(num_time_steps)])

        partial_dE_dv = dE_dz * python_spike_gradient(python_voltages, threshold_voltage)

        start = time.time()
        (python_dE_W_in,
         python_dE_W_rec,
         python_dE_tau_membrane) = python_backward_pass(samples, python_voltages, python_activations,
                                                        partial_dE_dv,
                                                        W_rec, tau_membrane,
                                                        threshold_voltage, dt,
                                                        dampening_factor=gradient_scaling_factor)

        python_backward_duration = time.time() - start

        # CUDA
        start = time.time()
        (cuda_voltages,
         cuda_activations) = rsnn.forward_pass(W_in_tf, W_rec_tf, tau_membrane_tf,
                                               samples_tf,
                                               threshold_voltage=threshold_voltage,
                                               delta_t=dt)
        cuda_forward_duration = time.time() - start

        smoothed_spikes = tf.stack(
            [tf.math.reduce_mean(cuda_activations[i - output_time_window: i], axis=0)
             if i >= output_time_window else tf.zeros(shape=[batch_size, num_neurons])
             for i in range(num_time_steps)])

        network_output = tf.linalg.matmul(smoothed_spikes, W_out_tf, transpose_b=True)

        softmax_output = tf.math.exp(network_output - tf.math.reduce_max(network_output))
        softmax_output = softmax_output / tf.math.reduce_sum(softmax_output, axis=-1, keepdims=True)

        indices_with_highest_probability = find_indices_of_max_probabilities(softmax_output)
        time_step_with_highest_prob_per_sample = indices_with_highest_probability[:, :2]

        predicted_distribution = tf.gather_nd(softmax_output, time_step_with_highest_prob_per_sample)

        dE_dnetwork_output_values = predicted_distribution - labels_tf
        dE_dnetwork_output = tf.scatter_nd(time_step_with_highest_prob_per_sample,
                                           dE_dnetwork_output_values,
                                           network_output.get_shape())

        dE_dsmoothed_spikes = tf.matmul(dE_dnetwork_output, W_out)

        dE_dz = tf.stack([tf.reduce_mean(dE_dsmoothed_spikes[i - output_time_window: i], axis=0)
                          if i >= output_time_window else tf.zeros(shape=[batch_size, num_neurons])
                          for i in range(num_time_steps)])

        partial_dE_dv = dE_dz * calculate_spike_gradient(cuda_voltages, threshold_voltage)

        start = time.time()

        (cuda_dE_dW_in,
         cuda_dE_dW_rec,
         cuda_dE_dtau_membrane) = rsnn.backward_pass(partial_dE_dv, W_rec_tf, tau_membrane_tf,
                                                     samples_tf,
                                                     cuda_voltages, cuda_activations,
                                                     threshold_voltage=threshold_voltage, delta_t=dt,
                                                     gradient_scaling_factor=gradient_scaling_factor)
        cuda_backward_duration = time.time() - start

        # Save timing results
        forward_pass_python.append(python_forward_duration)
        backward_pass_python.append(python_backward_duration)
        forward_pass_cuda.append(cuda_forward_duration)
        backward_pass_cuda.append(cuda_backward_duration)

    results[num_neurons] = {"python_forward": forward_pass_python,
                            "python_backward": backward_pass_python,
                            "cuda_forward": forward_pass_cuda,
                            "cuda_backward": backward_pass_cuda}

    if save_intermediate_results:
        with open(os.path.join(results_directory, f"run_time_results_{batch_size}_until_{num_neurons}.p"),
                  "wb") as pickle_file:
            pickle.dump(results, pickle_file)

hyperparameters = {"dt": dt,
                   "initial_membrane_time_constant": initial_membrane_time_constant,
                   "output_time_window": output_time_window,
                   "threshold_voltage": threshold_voltage,
                   "gradient_scaling_factor": gradient_scaling_factor}

info = {"num_repetitions": num_repetitions,
        "num_neurons": neuron_numbers_to_test,
        "batch_size": batch_size,
        "hyperparameters": hyperparameters}

for file_name in os.listdir(results_directory):
    if file_name.endswith(".p"):
        os.remove(os.path.join(results_directory, file_name))

with open(os.path.join(results_directory, f"final_run_time_results_{batch_size}.p"), "wb") as pickle_file:
    pickle.dump(results, pickle_file)

with open(os.path.join(results_directory, f"info.p"), "wb") as pickle_file:
    pickle.dump(info, pickle_file)

print("Experiment completed")
