import tensorflow as tf
import numpy as np

import time
import pickle
import os

from snn_utils import initialize_weights, initialize_data, convert_to_tensors
from snn_utils import python_forward_pass, MSE, python_backward_pass
from snn_utils import verify_forward_pass, print_voltage_discrepancies, save_to_pickle_files

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------


num_time_steps = 100
num_batches = 4
num_neurons = 16

num_input_channels = 2  # currently only supports 2
num_output_channels = 1

decay_factor = 0.95
threshold_voltage = 0.5

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------


input_weights, recurrent_weights, output_weights = initialize_weights(num_neurons,
                                                                      num_input_channels,
                                                                      num_output_channels)

time_series_data = initialize_data(num_time_steps,
                                   num_batches,
                                   num_input_channels)

tensor_conversions = convert_to_tensors(input_weights,
                                        recurrent_weights,
                                        output_weights,
                                        time_series_data)

input_weights_tensor, recurrent_weights_tensor, output_weights_tensor, time_series_data_tensor = tensor_conversions

# -------------------------------------------------------------------
# FORWARD PASS
# -------------------------------------------------------------------

spiking_module = tf.load_op_library("./spiking_network.so")

start = time.time()
resulting_voltages_tensor, resulting_activations_tensor = spiking_module.forward_pass(input_weights_tensor,
                                                                                      recurrent_weights_tensor,
                                                                                      time_series_data_tensor,
                                                                                      decay_factor=decay_factor,
                                                                                      threshold_voltage=threshold_voltage)

op_duration = time.time() - start

resulting_voltages = resulting_voltages_tensor.numpy()
resulting_activations = resulting_activations_tensor.numpy()

# Python implementation to check against:

start = time.time()

expected_voltages, expected_activations = python_forward_pass(input_weights, recurrent_weights,
                                                              time_series_data,
                                                              decay_factor, threshold_voltage)

python_duration = time.time() - start

# -------------------------------------------------------------------
# VALIDATING THE RESULTS
# -------------------------------------------------------------------

validation_results = verify_forward_pass(expected_voltages,
                                         expected_activations,
                                         resulting_voltages,
                                         resulting_activations)

cuda_voltages_ok, cuda_activations_ok, python_voltages_ok, python_activations_ok, voltages_match, activations_match = validation_results


print("Forward Pass Results: ")

if cuda_voltages_ok:

    print("Membrane voltages:")
    print(resulting_voltages[:, 0])

    print("\n--------------------------\n")

    print("Neuron activations")
    print(resulting_activations[:, 0])

    print("Activated voltages")

    print("\n--------------------------\n")

    print(resulting_voltages[:, 0][resulting_activations[:, 0].astype(bool)])

else:
    print('Voltages do not match within batches for each neuron')

print("\nCUDA Implementation:")
print(f"Are all voltages close to equal between batches? {cuda_voltages_ok}")
print(f"Are all activations close to equal between batches? {cuda_activations_ok}")

print("\nPython Implementation")
print(f"Are all voltages close to equal between batches? {python_voltages_ok}")
print(f"Are all activations close to equal between batches? {python_activations_ok}")

print("\n Comparing implementations:")
print(f"Do the membrane voltages match? {voltages_match}")
print(f"Do the neuron activations match between Python and CUDA? {activations_match}")

if not voltages_match:
    print_voltage_discrepancies(expected_voltages, resulting_voltages)

print(f"\nRuntime CUDA OP: {op_duration}")
print(f"Runtime native Python: {python_duration}")

save_data = True
if save_data:
    hyperparameters = np.array((num_time_steps,
                                num_batches,
                                num_neurons,
                                num_input_channels,
                                num_output_channels,
                                decay_factor,
                                threshold_voltage))

    save_to_pickle_files(hyperparameters,
                         input_weights, recurrent_weights, output_weights,
                         time_series_data,
                         resulting_voltages, resulting_activations)


# -------------------------------------------------------------------
# BACKWARD PASS
# -------------------------------------------------------------------
expected_output = (time_series_data[:, :, 0] + time_series_data[:, :, 1]).reshape(num_time_steps,
                                                                                  num_batches,
                                                                                  num_output_channels)

network_output = np.dot(resulting_voltages, output_weights.T)

dE_dy = -2 * (expected_output - network_output)
partial_dy_dv = output_weights
partial_dE_dv = np.dot(dE_dy, partial_dy_dv)

expected_dE_dW_in, expected_dE_dW_rec = python_backward_pass(time_series_data, resulting_voltages, resulting_activations,
                                                             partial_dE_dv, recurrent_weights,
                                                             threshold_voltage, decay_factor)

print("Input weights:")
print(f"Shape: {expected_dE_dW_in.shape}")
print("Result:")
print(expected_dE_dW_in)
print("")

print("Recurrent weights:")
print(f"Shape: {expected_dE_dW_rec.shape}")
print("Result:")
print(expected_dE_dW_rec)
