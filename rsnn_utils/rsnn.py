import numpy as np
import tensorflow as tf


def initialize_weights(num_neurons, num_input_channels, num_output_channels,
                       threshold_voltage, initial_membrane_time_constant):
    # normalized Xavier initializtion
    input_weights_limit = np.sqrt(6) / np.sqrt(num_neurons + num_input_channels)
    input_weights = np.random.uniform(-input_weights_limit, input_weights_limit, size=(num_neurons, num_input_channels))

    recurrent_weights_limit = np.sqrt(6) / np.sqrt(num_neurons + num_neurons)
    recurrent_weights = np.random.uniform(-recurrent_weights_limit, recurrent_weights_limit,
                                          size=(num_neurons, num_neurons))
    np.fill_diagonal(recurrent_weights, - threshold_voltage)

    output_weights_limit = np.sqrt(6) / np.sqrt(num_output_channels + num_neurons)
    output_weights = np.random.uniform(-output_weights_limit, output_weights_limit,
                                       size=(num_output_channels, num_neurons))

    membrane_decay_factors = np.ones((num_neurons, 1)) * initial_membrane_time_constant

    return input_weights, recurrent_weights, output_weights, membrane_decay_factors


def initialize_data(num_time_steps, num_batches, num_input_channels, start_value, end_value):

    if num_input_channels % 2 != 0:
        num_input_channels += 1

    x = np.linspace(start_value, end_value, num_time_steps)
    time_series_data = np.array([np.sin(2 / 3 * x), np.cos(x)]).T
    time_series_data_batch = time_series_data.reshape((num_time_steps, 1, 2))
    time_series_data_batch = np.repeat(time_series_data_batch, int(num_input_channels/2), axis=2).astype(np.single)
    time_series_data_batch = np.repeat(time_series_data_batch, num_batches, axis=1).astype(np.single)

    return time_series_data_batch


def convert_to_tensors(input_weights, recurrent_weights, output_weights, time_series_data, membrane_time_constants):

    input_weights_tensor = tf.convert_to_tensor(input_weights, dtype=float)
    recurrent_weights_tensor = tf.convert_to_tensor(recurrent_weights, dtype=float)
    output_weights_tensor = tf.convert_to_tensor(output_weights, dtype=float)
    time_series_data_tensor = tf.convert_to_tensor(time_series_data, dtype=float)
    membrane_time_constants_tensor = tf.convert_to_tensor(membrane_time_constants, dtype=float)

    return (input_weights_tensor, recurrent_weights_tensor, output_weights_tensor,
            time_series_data_tensor, membrane_time_constants_tensor)


def convert_batch_to_tensors(batched_data):

    batched_data_as_tensors = []

    for current_batch in batched_data:
        current_batch_tensor = tf.convert_to_tensor(current_batch, dtype=float)
        batched_data_as_tensors.append(current_batch_tensor)

    return batched_data_as_tensors


def python_forward_pass(input_weights, recurrent_weights, membrane_time_constants,
                        time_series_data, threshold_voltage, dt):

    num_time_steps, num_batches, num_input_channels = time_series_data.shape
    num_neurons, _ = recurrent_weights.shape

    v = np.zeros((num_batches, num_neurons))
    z = np.zeros((num_batches, num_neurons))

    membrane_decay_factors = np.exp(-dt/membrane_time_constants)

    expected_voltages = np.zeros((num_time_steps, num_batches, num_neurons))
    expected_activities = np.zeros((num_time_steps, num_batches, num_neurons))
    base_activity = np.dot(time_series_data, input_weights.T)

    for t in range(num_time_steps):
        v = membrane_decay_factors.T * v + base_activity[t] + np.dot(z, recurrent_weights.T)
        z[v >= threshold_voltage] = 1.0
        z[v < threshold_voltage] = 0.0

        expected_voltages[t] = v
        expected_activities[t] = z

    return expected_voltages, expected_activities


def MSE(expected_output, network_output):
    num_time_steps, num_batches, num_output_channels = network_output.shape

    difference = (expected_output - network_output) ** 2

    return np.sum(difference, axis=0) / num_time_steps


def spike_gradient(membrane_voltages, threshold_voltage, dampening_factor=0.3):

    return dampening_factor * np.maximum(1 - np.abs(membrane_voltages - threshold_voltage) / threshold_voltage, 0.0)


def calculate_spike_gradient(membrane_voltages, threshold_voltage, dampening_factor=0.3):
    return dampening_factor * tf.math.maximum(1 - tf.math.abs(membrane_voltages - threshold_voltage) / threshold_voltage, 0.0)


def old_python_backward_pass(time_series_data, resulting_voltages, resulting_activations,
                         partial_dE_dv,
                         recurrent_weights, membrane_time_constants,
                         threshold_voltage, dt, dampening_factor=0.3):

    num_time_steps, num_batches, num_input_channels = time_series_data.shape
    *_, num_neurons = resulting_voltages.shape

    previous_total_dE_dv = np.zeros((num_batches, num_neurons))
    sum_over_k = np.zeros((num_batches, num_neurons))
    membrane_decay_factors = np.exp(-dt/membrane_time_constants)

    dE_dW_in = np.zeros((num_neurons, num_input_channels))
    dE_dW_rec = np.zeros((num_neurons, num_neurons))
    dE_dalpha = np.zeros((num_neurons, 1))

    # all_total_dE_dv = np.zeros((num_time_steps, num_batches, num_neurons))
    # membrane_decay_components = np.zeros((num_time_steps, 1, num_neurons))
    # input_components = np.zeros((num_time_steps, num_neurons, num_input_channels))
    # recurrent_components = np.zeros((num_time_steps, num_neurons, num_neurons))
    # next_voltages_tensor = np.zeros((num_time_steps, num_batches, num_neurons))

    for time_step in reversed(range(num_time_steps)):
        current_membrane_voltages = resulting_voltages[time_step]
        spike_gradient_approximate = spike_gradient(current_membrane_voltages, threshold_voltage,
                                                    dampening_factor=dampening_factor)

        for batch in range(num_batches):
            batchwise_spike_gradients = spike_gradient_approximate[batch]
            batchwise_dv_k_dv_j = batchwise_spike_gradients.reshape(1, -1) * recurrent_weights + np.diag(membrane_decay_factors[:, 0])

            batchwise_previous_total_dE_dv = previous_total_dE_dv[batch].reshape(1, -1)
            sum_over_k[batch] = np.dot(batchwise_previous_total_dE_dv, batchwise_dv_k_dv_j)

        current_partial_dE_dv = partial_dE_dv[time_step]
        current_total_dE_dv = current_partial_dE_dv + sum_over_k

        dE_dW_in_component = np.dot(current_total_dE_dv.T, time_series_data[time_step])
        dE_dW_rec_component = np.dot(current_total_dE_dv.T, resulting_activations[time_step])

        try:
            if time_step > 0:
                next_membrane_voltages = resulting_voltages[time_step - 1]
                dE_dalpha_component = np.sum(current_total_dE_dv * next_membrane_voltages,
                                             axis=0,
                                             keepdims=True)
            else:
                dE_dalpha_component = np.zeros((1, num_neurons))

        except:
            print(time_step)
            print(current_total_dE_dv)
            print(next_membrane_voltages)
            print(dE_dalpha_component)

            raise ValueError("hi")

        dE_dW_in += dE_dW_in_component
        dE_dW_rec += dE_dW_rec_component
        dE_dalpha += dE_dalpha_component.T

        previous_total_dE_dv = current_total_dE_dv

        # all_total_dE_dv[time_step] = current_total_dE_dv
        # input_components[time_step] = dE_dW_in_component
        # recurrent_components[time_step] = dE_dW_rec_component
        # next_voltages_tensor[time_step] = next_membrane_voltages
        # membrane_decay_components[time_step] = dE_dalpha_component

    dE_dmembrane_time_constants = (dE_dalpha * dt * membrane_decay_factors) / (
                membrane_time_constants * membrane_time_constants)

    return dE_dW_in, dE_dW_rec, dE_dmembrane_time_constants


def python_backward_pass(time_series_data, resulting_voltages, resulting_activations,
                         partial_dE_dv,
                         recurrent_weights, membrane_time_constants,
                         threshold_voltage, dt, dampening_factor=0.3):

    num_time_steps, num_batches, num_input_channels = time_series_data.shape
    *_, num_neurons = resulting_voltages.shape

    previous_total_dE_dv = np.zeros((num_batches, num_neurons))
    sum_over_k = np.zeros((num_batches, num_neurons))
    membrane_decay_factors = np.exp(-dt/membrane_time_constants)

    dE_dW_in = np.zeros((num_neurons, num_input_channels))
    dE_dW_rec = np.zeros((num_neurons, num_neurons))
    dE_dalpha = np.zeros((num_neurons, 1))

    # all_total_dE_dv = np.zeros((num_time_steps, num_batches, num_neurons))
    # membrane_decay_components = np.zeros((num_time_steps, 1, num_neurons))
    # input_components = np.zeros((num_time_steps, num_neurons, num_input_channels))
    # recurrent_components = np.zeros((num_time_steps, num_neurons, num_neurons))
    # next_voltages_tensor = np.zeros((num_time_steps, num_batches, num_neurons))

    for time_step in reversed(range(num_time_steps)):
        current_membrane_voltages = resulting_voltages[time_step]
        spike_gradient_approximate = spike_gradient(current_membrane_voltages, threshold_voltage,
                                                    dampening_factor=dampening_factor)

        dv_k_dv_j = np.einsum("bn,mn->bmn", spike_gradient_approximate, recurrent_weights) + np.diag(membrane_decay_factors[:, 0])[
                                                                                 None, :, :]
        sum_over_k = np.einsum("bn,bnm->bm", previous_total_dE_dv, dv_k_dv_j)

        current_partial_dE_dv = partial_dE_dv[time_step]
        current_total_dE_dv = current_partial_dE_dv + sum_over_k

        dE_dW_in_component = np.dot(current_total_dE_dv.T, time_series_data[time_step])
        dE_dW_rec_component = np.dot(current_total_dE_dv.T, resulting_activations[time_step])

        try:
            if time_step > 0:
                next_membrane_voltages = resulting_voltages[time_step - 1]
                dE_dalpha_component = np.sum(current_total_dE_dv * next_membrane_voltages,
                                             axis=0,
                                             keepdims=True)
            else:
                dE_dalpha_component = np.zeros((1, num_neurons))

        except:
            print(time_step)
            print(current_total_dE_dv)
            print(next_membrane_voltages)
            print(dE_dalpha_component)

            raise ValueError("hi")

        dE_dW_in += dE_dW_in_component
        dE_dW_rec += dE_dW_rec_component
        dE_dalpha += dE_dalpha_component.T

        previous_total_dE_dv = current_total_dE_dv

        # all_total_dE_dv[time_step] = current_total_dE_dv
        # input_components[time_step] = dE_dW_in_component
        # recurrent_components[time_step] = dE_dW_rec_component
        # next_voltages_tensor[time_step] = next_membrane_voltages
        # membrane_decay_components[time_step] = dE_dalpha_component

    dE_dmembrane_time_constants = (dE_dalpha * dt * membrane_decay_factors) / (
                membrane_time_constants * membrane_time_constants)

    return dE_dW_in, dE_dW_rec, dE_dmembrane_time_constants