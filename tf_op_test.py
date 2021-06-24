import tensorflow as tf
import numpy as np

recurrent_weights = np.random.randn(4, 4) * 0.05
np.fill_diagonal(recurrent_weights, 0.0)
recurrent_weights_tensor = tf.convert_to_tensor(recurrent_weights)

input_weights = np.random.randn(2, 4) * 0.1
input_weights_tensor = tf.convert_to_tensor(input_weights)

x = np.arange(20)
time_series_data = np.array([np.sin(np.pi*x/10), np.cos(np.pi*x/20)]).T
time_series_data_tensor = tf.convert_to_tensor(time_series_data)

decay_factor = 0.95
threshold_voltage = 0.5

spiking_module = tf.load_op_library("./spiking_network.so")

resulting_voltages, resulting_activities = spiking_module.forward_pass(input_weights_tensor,
                                                                       recurrent_weights_tensor,
                                                                       time_series_data_tensor,
                                                                       decay_factor,
                                                                       threshold_voltage)

print("Membrane voltages:")
print(resulting_voltages.numpy())

print("\n--------------------------\n")

print("Neuron activities")
print(resulting_activities.numpy())
