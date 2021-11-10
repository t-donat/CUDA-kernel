import numpy as np
import tensorflow as tf
import pickle
import os


def read_pickle_files(file_directory="./data"):
    with open(os.path.join(file_directory, "hyperparameters.p"), "rb") as pickle_file:
        hyperparameters = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "input_weights.p"), "rb") as pickle_file:
        input_weights = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "recurrent_weights.p"), "rb") as pickle_file:
        recurrent_weights = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "output_weights.p"), "rb") as pickle_file:
        output_weights = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "time_series_data.p"), "rb") as pickle_file:
        time_series_data = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "resulting_voltages.p"), "rb") as pickle_file:
        resulting_voltages = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "resulting_activations.p"), "rb") as pickle_file:
        resulting_activations = pickle.load(pickle_file)

    return (hyperparameters,
            input_weights, recurrent_weights, output_weights,
            time_series_data,
            resulting_voltages, resulting_activations)


def save_to_pickle_files(network_hyperparameters,
                         input_weights, recurrent_weights, output_weights,
                         time_series_data,
                         resulting_voltages, resulting_activations,
                         file_directory="./data/pickled_data"):

    # write data to output files
    os.makedirs(file_directory,
                exist_ok=True)

    with open(os.path.join(file_directory, "hyperparameters.p"), "wb") as pickle_file:
        pickle.dump(network_hyperparameters, pickle_file)

    with open(os.path.join(file_directory, "input_weights.p"), "wb") as pickle_file:
        pickle.dump(input_weights, pickle_file)

    with open(os.path.join(file_directory, "recurrent_weights.p"), "wb") as pickle_file:
        pickle.dump(recurrent_weights, pickle_file)

    with open(os.path.join(file_directory, "output_weights.p"), "wb") as pickle_file:
        pickle.dump(output_weights, pickle_file)

    with open(os.path.join(file_directory, "time_series_data.p"), "wb") as pickle_file:
        pickle.dump(time_series_data, pickle_file)

    with open(os.path.join(file_directory, "resulting_voltages.p"), "wb") as pickle_file:
        pickle.dump(resulting_voltages, pickle_file)

    with open(os.path.join(file_directory, "resulting_activations.p"), "wb") as pickle_file:
        pickle.dump(resulting_activations, pickle_file)

    print("\nSuccessfully saved to pickle files!")


def initialize_weights(num_neurons, num_input_channels, num_output_channels):

    input_weights = np.random.randn(num_neurons, num_input_channels).astype(np.single) * 0.1

    recurrent_weights = np.random.randn(num_neurons, num_neurons).astype(np.single) * 0.1
    np.fill_diagonal(recurrent_weights, 0.0)

    output_weights = np.ones((num_output_channels, num_neurons)).astype(np.single)

    return input_weights, recurrent_weights, output_weights


def initialize_data(num_time_steps, num_batches, num_input_channels):
    x = np.linspace(0, 2 * np.pi, num_time_steps)
    time_series_data = np.array([np.sin(2 / 3 * x), np.cos(x)]).T
    time_series_data_batch = time_series_data.reshape((num_time_steps, 1, num_input_channels))
    time_series_data_batch = np.repeat(time_series_data_batch, num_batches, axis=1).astype(np.single)

    return time_series_data_batch


def convert_to_tensors(input_weights, recurrent_weights, output_weights, time_series_data):

    input_weights_tensor = tf.convert_to_tensor(input_weights, dtype=float)
    recurrent_weights_tensor = tf.convert_to_tensor(recurrent_weights, dtype=float)
    output_weights_tensor = tf.convert_to_tensor(output_weights, dtype=float)
    time_series_data_tensor = tf.convert_to_tensor(time_series_data, dtype=float)

    return input_weights_tensor, recurrent_weights_tensor, output_weights_tensor, time_series_data_tensor


def python_forward_pass(input_weights, recurrent_weights,
                        time_series_data,
                        decay_factor, threshold_voltage):

    num_time_steps, num_batches, num_input_channels = time_series_data.shape
    num_neurons, _ = recurrent_weights.shape

    v = np.zeros((num_batches, num_neurons))
    z = np.zeros((num_batches, num_neurons))

    expected_voltages = np.zeros((num_time_steps, num_batches, num_neurons))
    expected_activities = np.zeros((num_time_steps, num_batches, num_neurons))
    base_activity = np.dot(time_series_data, input_weights.T)

    for t in range(num_time_steps):
        current_input = base_activity[t]

        v = decay_factor * v + current_input + np.dot(z, recurrent_weights.T) - threshold_voltage * z
        z[v >= threshold_voltage] = 1.0
        z[v < threshold_voltage] = 0.0

        expected_voltages[t] = v
        expected_activities[t] = z

    return expected_voltages, expected_activities


def MSE(expected_output, network_output):
    num_time_steps, num_batches, num_output_channels = network_output.shape

    difference = (expected_output - network_output) ** 2

    return np.sum(difference, axis=0) / num_time_steps


def spike_gradient(membrane_voltages, threshold_voltage, dampening_factor):

    return dampening_factor * np.maximum(1 - np.abs(membrane_voltages - threshold_voltage) / threshold_voltage, 0.0)


def python_backward_pass(time_series_data, resulting_voltages, resulting_activations,
                         partial_dE_dv,
                         recurrent_weights,
                         threshold_voltage, decay_factor, dampening_factor):

    num_time_steps, num_batches, num_input_channels = time_series_data.shape
    *_, num_neurons = resulting_voltages.shape

    previous_total_dE_dv = np.zeros((num_batches, num_neurons))
    sum_over_k = np.zeros((num_batches, num_neurons))

    dE_dW_in = np.zeros((num_neurons, num_input_channels))
    dE_dW_rec = np.zeros((num_neurons, num_neurons))

    all_total_dE_dv = np.zeros((num_time_steps, num_batches, num_neurons))

    for time_step in reversed(range(num_time_steps)):
        current_membrane_voltages = resulting_voltages[time_step]
        spike_gradient_approximate = spike_gradient(current_membrane_voltages, threshold_voltage, dampening_factor)

        for batch in range(num_batches):
            batchwise_spike_gradients = spike_gradient_approximate[batch]
            batchwise_dv_k_dv_j = batchwise_spike_gradients.reshape(1, -1) * recurrent_weights + np.diag(
                decay_factor - threshold_voltage * batchwise_spike_gradients)

            batchwise_previous_total_dE_dv = previous_total_dE_dv[batch].reshape(1, -1)
            sum_over_k[batch] = np.dot(batchwise_previous_total_dE_dv, batchwise_dv_k_dv_j)

        current_partial_dE_dv = partial_dE_dv[time_step]
        current_total_dE_dv = current_partial_dE_dv + sum_over_k

        all_total_dE_dv[time_step] = current_total_dE_dv

        dE_dW_in_component = np.dot(current_total_dE_dv.T, time_series_data[time_step])
        dE_dW_rec_component = np.dot(current_total_dE_dv.T, resulting_activations[time_step])

        dE_dW_in += dE_dW_in_component
        dE_dW_rec += dE_dW_rec_component

        previous_total_dE_dv = current_total_dE_dv

    return dE_dW_in, dE_dW_rec, all_total_dE_dv


def verify_forward_pass(expected_voltages, expected_activations, calculated_voltages, calculated_activations):

    cuda_voltages_ok = all([np.allclose(time_step_data, time_step_data[0]) for time_step_data in calculated_voltages])
    cuda_activities_ok = all([np.allclose(time_step_data, time_step_data[0]) for time_step_data in calculated_activations])

    python_voltages_ok = np.allclose(expected_voltages, expected_voltages[:, 0, None])
    python_activations_ok = np.allclose(expected_activations, expected_activations[:, 0, None])

    voltages_match = np.allclose(expected_voltages, calculated_voltages)
    activities_match = np.allclose(expected_activations, calculated_activations)

    return cuda_voltages_ok, cuda_activities_ok, python_voltages_ok, python_activations_ok, voltages_match, activities_match


def print_voltage_discrepancies(expected_voltages, calculated_voltages):
    print("\nMembrane voltage discrepancies (larger than 10⁻⁸):\n")

    num_time_steps, num_batches, num_neurons = expected_voltages.shape

    differences = expected_voltages - calculated_voltages[:, 0].reshape(num_time_steps, 1, num_neurons)
    different_spots = np.invert(np.isclose(expected_voltages, calculated_voltages[:, 0].reshape(num_time_steps,
                                                                                                1,
                                                                                                num_neurons)))

    discrepancy_values = np.abs(differences.transpose(0, 2, 1)[different_spots.transpose(0, 2, 1)])
    # to avoid repeating the same difference that is shared across the batches
    duplicates_removed = discrepancy_values[::num_batches]

    orders_of_magnitude = np.floor(np.log10(np.abs(duplicates_removed))).astype(int)
    unique_orders_of_magnitude, magnitude_counts = np.unique(orders_of_magnitude, return_counts=True)

    num_discrepancies = len(duplicates_removed)
    num_values_per_batch = different_spots[:, 0].size

    print('Order of Magnitude \tNumber of samples \tPercentage of discrepancies')
    for current_magnitude, current_counts in zip(unique_orders_of_magnitude, magnitude_counts):
        print(f"{current_magnitude} \t\t\t{current_counts} \t\t\t{round(current_counts / num_discrepancies * 100, 2)}%")

    print(
        f"\nTotal number of discrepancies: {num_discrepancies} ({round(num_discrepancies / num_values_per_batch * 100, 2)}% of all values)", )


def verify_backward_pass(expected_input_gradient, expected_recurrent_gradient,
                         calculated_input_gradient, calculated_recurrent_gradient):

    input_weights_match = np.allclose(expected_input_gradient, calculated_input_gradient)
    recurrent_weights_match = np.allclose(expected_recurrent_gradient, calculated_recurrent_gradient)

    return input_weights_match, recurrent_weights_match

def print_weight_discrepancies():
    pass
