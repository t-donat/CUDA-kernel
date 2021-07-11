import tensorflow as tf
import numpy as np

num_timesteps = 50
num_batches = 5
num_input_channels = 2
num_neurons = 4

recurrent_weights = np.random.randn(num_neurons, num_neurons) * 0.1
np.fill_diagonal(recurrent_weights, 0.0)
recurrent_weights_tensor = tf.convert_to_tensor(recurrent_weights, dtype=float)

input_weights = np.random.randn(num_input_channels, num_neurons) * 0.1
input_weights_tensor = tf.convert_to_tensor(input_weights, dtype=float)

x = np.arange(num_timesteps) * 2 / num_timesteps
time_series_data = np.array([np.sin(2 / 3 * np.pi*x), np.cos(np.pi*x)]).T
time_series_data_batch = time_series_data.reshape((num_timesteps, 1, num_input_channels))
time_series_data_batch = np.repeat(time_series_data_batch, num_batches, axis=1)
time_series_data_tensor = tf.convert_to_tensor(time_series_data_batch, dtype=float)

decay_factor = 0.95
threshold_voltage = 0.5

spiking_module = tf.load_op_library("./spiking_network.so")

resulting_voltages, resulting_activities = spiking_module.forward_pass(input_weights_tensor,
                                                                       recurrent_weights_tensor,
                                                                       time_series_data_tensor,
                                                                       decay_factor=decay_factor,
                                                                       threshold_voltage=threshold_voltage)

resulting_voltages_array = resulting_voltages.numpy()
resulting_activities_array = resulting_activities.numpy()

print("results: ")

for batch in range(num_batches):
    print("Membrane voltages:")
    print(resulting_voltages_array[:, 0])

    print("\n--------------------------\n")

    print("Neuron activities")
    print(resulting_activities_array[:, 0])


    print("Activated voltages")

    print("\n--------------------------\n")

    print(resulting_voltages_array[:, 0][resulting_activities_array[:, 0].astype(bool)])

print(f"All voltage batches close to equal: {all([np.allclose(timestep, timestep[0]) for timestep in resulting_voltages_array])}")
print(f"All activity batches close to equal: {all([np.allclose(timestep, timestep[0]) for timestep in resulting_activities_array])}")

# Python implementation to check against:

base_activity = np.dot(time_series_data, input_weights)

v = np.zeros((num_neurons, 1))
z = np.zeros((num_neurons, 1))

expected_voltages = np.zeros((num_timesteps, num_neurons))
expected_activities = np.zeros((num_timesteps, num_neurons))

for t in range(num_timesteps):
    current_input = base_activity[t][:, None]

    v = decay_factor * v + current_input + np.dot(recurrent_weights.T, z) - threshold_voltage * z
    z[v >= threshold_voltage] = 1.0
    z[v < threshold_voltage] = 0.0

    expected_voltages[t] = v.flatten()
    expected_activities[t] = z.flatten()

voltages_close = np.allclose(expected_voltages, resulting_voltages_array[:, 0], atol=1e-6)
print(f"Do the voltages match: {voltages_close}")

if not voltages_close:
    not_close = (expected_voltages - resulting_voltages_array[:, 0])[np.invert(np.isclose(expected_voltages,
                                                                                          resulting_voltages_array[:,
                                                                                                                   0]))]
    print("Differences:")
    print(not_close)


print(f"Do the activities match: {np.allclose(expected_activities, resulting_activities_array[:, 0])}")
