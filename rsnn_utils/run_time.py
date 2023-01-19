import time
import numpy as np
import tensorflow as tf

from rsnn_utils.rsnn import python_forward_pass, python_backward_pass, python_spike_gradient, calculate_spike_gradient
from rsnn_utils.data import find_indices_of_max_probabilities


def time_python_implementation(samples_of_test, labels_of_test, W_in, W_rec, tau_membrane, W_out, hyperparameters):

    # Unpack hyperparameters
    dt = hyperparameters.dt
    threshold_voltage = hyperparameters.threshold_voltage
    output_time_window = hyperparameters.output_time_window
    num_neurons = hyperparameters.num_neurons
    gradient_scaling_factor = hyperparameters.gradient_scaling_factor

    num_time_steps, batch_size, _ = samples_of_test.shape

    # Forward Pass
    start = time.time()
    resulting_voltages, resulting_activations = python_forward_pass(W_in, W_rec, tau_membrane,
                                                                    samples_of_test,
                                                                    threshold_voltage, dt)
    forward_duration = time.time() - start

    smoothed_spikes = np.stack([np.mean(resulting_activations[i - output_time_window:i], axis=0)
                                if i >= output_time_window else np.zeros((batch_size, num_neurons))
                                for i in range(num_time_steps)])

    network_output = np.dot(smoothed_spikes, W_out.T)

    softmax_output = np.exp(network_output - np.max(network_output))
    softmax_output = softmax_output / np.sum(softmax_output, axis=-1, keepdims=True)

    # Gradient Calculation
    dE_dnetwork_output = np.zeros_like(network_output)

    for b in range(batch_size):
        current_data = softmax_output[:, b]
        argmax_index = np.argmax(current_data)

        argmax_time_step = argmax_index // num_neurons
        argmax_class = argmax_index % num_neurons

        predicted_distribution = current_data[argmax_time_step]
        ground_truth_distribution = labels_of_test[b]

        dE_dnetwork_output[argmax_time_step, b] = predicted_distribution - ground_truth_distribution

    dE_dsmoothed_spikes = np.dot(dE_dnetwork_output, W_out)

    dE_dz = np.stack([np.mean(dE_dsmoothed_spikes[i - output_time_window:i], axis=0)
                      if i >= output_time_window else np.zeros((batch_size, num_neurons))
                      for i in range(num_time_steps)])

    partial_dE_dv = dE_dz * python_spike_gradient(resulting_voltages, threshold_voltage)

    # Backward Pass
    start = time.time()
    (dE_W_in,
     dE_W_rec,
     dE_tau_membrane) = python_backward_pass(samples_of_test, resulting_voltages, resulting_activations,
                                             partial_dE_dv,
                                             W_rec, tau_membrane,
                                             threshold_voltage, dt,
                                             dampening_factor=gradient_scaling_factor)

    backward_duration = time.time() - start

    return forward_duration, backward_duration


def time_cuda_implementation(samples_of_test, labels_of_test,
                             W_in, W_rec, tau_membrane, W_out,
                             hyperparameters, cuda_source_library):

    # Unpack hyperparameters
    dt = hyperparameters.dt
    threshold_voltage = hyperparameters.threshold_voltage
    output_time_window = hyperparameters.output_time_window
    num_neurons = hyperparameters.num_neurons
    gradient_scaling_factor = hyperparameters.gradient_scaling_factor

    num_time_steps, batch_size, _ = samples_of_test.shape

    # Forward Pass
    start = time.time()
    resulting_voltages, resulting_activations = cuda_source_library.forward_pass(W_in, W_rec, tau_membrane,
                                                                                 samples_of_test,
                                                                                 threshold_voltage=threshold_voltage,
                                                                                 delta_t=dt)
    forward_duration = time.time() - start

    smoothed_spikes = tf.stack(
        [tf.math.reduce_mean(resulting_activations[i - output_time_window: i], axis=0)
         if i >= output_time_window else tf.zeros(shape=[batch_size, num_neurons])
         for i in range(num_time_steps)])

    network_output = tf.linalg.matmul(smoothed_spikes, W_out, transpose_b=True)

    softmax_output = tf.math.exp(network_output - tf.math.reduce_max(network_output))
    softmax_output = softmax_output / tf.math.reduce_sum(softmax_output, axis=-1, keepdims=True)

    indices_with_highest_probability = find_indices_of_max_probabilities(softmax_output)
    time_step_with_highest_prob_per_sample = indices_with_highest_probability[:, :2]

    predicted_distribution = tf.gather_nd(softmax_output, time_step_with_highest_prob_per_sample)

    # Gradient Calculation
    dE_dnetwork_output_values = predicted_distribution - labels_of_test
    dE_dnetwork_output = tf.scatter_nd(time_step_with_highest_prob_per_sample,
                                       dE_dnetwork_output_values,
                                       network_output.get_shape())

    dE_dsmoothed_spikes = tf.matmul(dE_dnetwork_output, W_out)

    dE_dz = tf.stack([tf.reduce_mean(dE_dsmoothed_spikes[i - output_time_window: i], axis=0)
                      if i >= output_time_window else tf.zeros(shape=[batch_size, num_neurons])
                      for i in range(num_time_steps)])

    partial_dE_dv = dE_dz * calculate_spike_gradient(resulting_voltages, threshold_voltage)

    # Backward Pass
    start = time.time()

    (dE_dW_in,
     dE_dW_rec,
     dE_dtau_membrane) = cuda_source_library.backward_pass(partial_dE_dv, W_rec, tau_membrane,
                                                           samples_of_test,
                                                           resulting_voltages, resulting_activations,
                                                           threshold_voltage=threshold_voltage, delta_t=dt,
                                                           gradient_scaling_factor=gradient_scaling_factor)
    backward_duration = time.time() - start

    return forward_duration, backward_duration
