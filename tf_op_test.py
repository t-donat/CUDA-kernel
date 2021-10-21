import tensorflow as tf
import numpy as np

import time
import pickle
import os


def spike_gradient(membrane_voltages, threshold_voltage):
    dampening_factor = 0.3

    return dampening_factor * np.maximum(1 - np.abs(membrane_voltages - threshold_voltage) / v_th, 0.0)


num_timesteps = 1000
num_batches = 128
num_input_channels = 2
num_neurons = 128

recurrent_weights = np.random.randn(num_neurons, num_neurons) * 0.1
np.fill_diagonal(recurrent_weights, 0.0)
recurrent_weights_tensor = tf.convert_to_tensor(recurrent_weights, dtype=float)

input_weights = np.random.randn(num_neurons, num_input_channels) * 0.1
input_weights_tensor = tf.convert_to_tensor(input_weights, dtype=float)

x = np.arange(num_timesteps) * 2 / num_timesteps
time_series_data = np.array([np.sin(2 / 3 * np.pi*x), np.cos(np.pi*x)]).T
time_series_data_batch = time_series_data.reshape((num_timesteps, 1, num_input_channels))
time_series_data_batch = np.repeat(time_series_data_batch, num_batches, axis=1)
time_series_data_tensor = tf.convert_to_tensor(time_series_data_batch, dtype=float)

decay_factor = 0.95
threshold_voltage = 0.5

spiking_module = tf.load_op_library("./spiking_network.so")

start = time.time()
resulting_voltages, resulting_activities = spiking_module.forward_pass(input_weights_tensor,
                                                                       recurrent_weights_tensor,
                                                                       time_series_data_tensor,
                                                                       decay_factor=decay_factor,
                                                                       threshold_voltage=threshold_voltage)

op_duration = time.time() - start

resulting_voltages_array = resulting_voltages.numpy()
resulting_activities_array = resulting_activities.numpy()

print("results: ")

batch_voltages_ok = all([np.allclose(timestep, timestep[0]) for timestep in resulting_voltages_array])
batch_activities_ok = all([np.allclose(timestep, timestep[0]) for timestep in resulting_activities_array])

# Python implementation to check against:

start = time.time()
base_activity = np.dot(time_series_data_batch, input_weights.T)

v = np.zeros((num_batches, num_neurons))
z = np.zeros((num_batches, num_neurons))

expected_voltages = np.zeros((num_timesteps, num_batches, num_neurons))
expected_activities = np.zeros((num_timesteps, num_batches, num_neurons))

for t in range(num_timesteps):
    current_input = base_activity[t]

    v = decay_factor * v + current_input + np.dot(z, recurrent_weights.T) - threshold_voltage * z
    z[v >= threshold_voltage] = 1.0
    z[v < threshold_voltage] = 0.0

    expected_voltages[t] = v
    expected_activities[t] = z

python_duration = time.time() - start

python_same_voltages = np.allclose(expected_voltages, expected_voltages[:, 0, None])
python_same_activities = np.allclose(expected_activities, expected_activities[:, 0, None])

voltages_ok = np.allclose(expected_voltages, resulting_voltages_array)
activities_ok = np.allclose(expected_activities, resulting_activities_array)

if python_same_voltages:

    print("Membrane voltages:")
    print(resulting_voltages_array[:, 0])

    print("\n--------------------------\n")

    print("Neuron activities")
    print(resulting_activities_array[:, 0])

    print("Activated voltages")

    print("\n--------------------------\n")

    print(resulting_voltages_array[:, 0][resulting_activities_array[:, 0].astype(bool)])

else:
    print('Voltages do not match within batches for each neuron')

print("\n")

if all([batch_activities_ok,
        batch_voltages_ok,
        voltages_ok,
        activities_ok]):

    print("All tests: OK")

else:

    print(f"Are all voltage batches close to equal? {batch_voltages_ok}")
    print(f"Are all activity batches close to equal? {batch_activities_ok}")

    print(f"Do the membrane voltages match between Python and CUDA? {voltages_ok}")
    print(f"Do the neuron activations match between Python and CUDA? {activities_ok}")

    if not voltages_ok:

        print("\nMembrane voltage discrepancies (larger than 10⁻⁸):\n")
        differences = expected_voltages - resulting_voltages_array[:, 0].reshape(num_timesteps, 1, num_neurons)
        different_spots = np.invert(np.isclose(expected_voltages, resulting_voltages_array[:, 0].reshape(num_timesteps,
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
            print(f"{current_magnitude} \t\t\t{current_counts} \t\t\t{round(current_counts/num_discrepancies * 100, 2)}%")

        print(f"\nTotal number of discrepancies: {num_discrepancies} ({round(num_discrepancies/num_values_per_batch * 100, 2)}% of all values)", )

        #while True:
        #    user_input = input('Should all voltage discrepancies be printed? (y/N): ')

        #    if user_input in ['N', 'n', '']:
        #        break # while loop

        #    elif user_input in ['Y', 'y']:


        #        time_step_indices, neuron_indices, batch_indices = np.array(np.where(different_spots))[:, ::num_batches]
        #        indices_of_different_spots = zip(time_step_indices, neuron_indices)

        #        print("\nDifference values:")

        #        for (curent_time_step_index, current_neuron_index), current_value_of_difference in zip(indices_of_different_spots, duplicates_removed):

        #            print(f"t: {curent_time_step_index}, b: all, n: {current_neuron_index}, absolute_difference: {current_value_of_difference}")
                    # print(f"t: {time_step_index}, b: {batch_index}, n: {neuron_index}, absolute_difference: {value_of_difference}")

        #        break # while loop

print(f"\nRuntime CUDA OP: {op_duration}")
print(f"Runtime native Python: {python_duration}")

if True:
    # write data to output files
    save_directory = "./data/pickled_data"
    os.makedirs(save_directory, exist_ok=True)

    hyperparameters = np.array((num_timesteps, num_batches, num_input_channels, num_neurons, decay_factor, threshold_voltage))

    with open(os.path.join(save_directory, "hyperparameters.p"), "wb") as pickle_file:
        pickle.dump(hyperparameters, pickle_file)

    with open(os.path.join(save_directory, "input_weights.p"), "wb") as pickle_file:
        pickle.dump(input_weights, pickle_file)

    with open(os.path.join(save_directory, "recurrent_weights.p"), "wb") as pickle_file:
        pickle.dump(recurrent_weights, pickle_file)

    with open(os.path.join(save_directory, "time_series_data.p"), "wb") as pickle_file:
        pickle.dump(time_series_data_batch, pickle_file)

    with open(os.path.join(save_directory, "resulting_voltages.p"), "wb") as pickle_file:
        pickle.dump(resulting_voltages_array, pickle_file)

    with open(os.path.join(save_directory, "resulting_activations.p"), "wb") as pickle_file:
        pickle.dump(resulting_activities_array, pickle_file)

    print("\nSuccessfully saved to pickle files!")

exit(0)
# -------------------------------------------------------------------
# BACKWARD PASS
# -------------------------------------------------------------------
dE_dv_partial = - 2 * (expected_output.flatten() - predicted_output.flatten()).reshape(-1, 1)

# dE_dv_total = np.zeros((num_timesteps, num_batches, num_neurons))
previous_dE_dv = np.zeros((num_batches, num_neurons))

dE_dW_in = np.zeros_like(W_in)
dE_dW_rec = np.zeros_like(W_rec)

for time_step in reversed(range(num_timesteps)):
    current_membrane_voltages = resulting_voltages_array[time_step]
    psi = spike_gradient(current_membrane_voltages, threshold_voltage)

    dvk_dvj = np.diag(decay_factor - v_th * psi.flatten()) + W_rec * psi.T

    current_dE_dv = dE_dv_partial[time_step] * np.ones((num_batches, num_neurons)) + np.sum(dvk_dvj * previous_dE_dv, axis=0)

    # implicit summation over batches due to the outer product
    dE_dW_in_component = np.dot(input_data[time_step].T, current_dE_dv)
    dE_dW_rec_component = np.dot(spike_activity[time_step].T, current_dE_dv)

    assert dE_dW_in_component.shape == W_in.shape
    assert dE_dW_rec_component.shape == W_rec.shape

    dE_dW_in += dE_dW_in_component
    dE_dW_rec += dE_dW_rec_component

    previous_dE_dv = current_dE_dv

print("Input weights:")
print(f"Shape: {dE_dW_in.shape}")
print("Result:")
print(dE_dW_in)
print("")

print("Recurrent weights:")
print(f"Shape: {dE_dW_rec.shape}")
print("Result:")
print(dE_dW_rec)



