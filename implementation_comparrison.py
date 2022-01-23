import tensorflow as tf
import numpy as np

import time
import pickle
import os

from rsnn_utils.io import save_to_pickle_files
from rsnn_utils.rsnn import initialize_weights, initialize_data, convert_to_tensors
from rsnn_utils.rsnn import python_forward_pass, python_backward_pass
from rsnn_utils.validation import verify_forward_pass, verify_backward_pass
from rsnn_utils.validation import print_voltage_discrepancies, print_weight_discrepancies

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

verbose = True
save_data = True
debug_mode = True

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

start_value = 0
end_value = 2 * np.pi
num_time_steps = 100

dt = (end_value - start_value)/num_time_steps
initial_membrane_time_constant = 0.02  # 20 ms

num_batches = 1
num_neurons = 8
num_input_channels = 2  # data initialization currently only supports 2 channels
num_output_channels = 1

decay_factor = np.exp(-dt/initial_membrane_time_constant)
threshold_voltage = 0.5
gradient_scaling_factor = 0.3

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------


(input_weights, recurrent_weights,
 output_weights, membrane_time_constants) = initialize_weights(num_neurons, num_input_channels, num_output_channels,
                                                              threshold_voltage, initial_membrane_time_constant)

membrane_time_constants_tensor = tf.convert_to_tensor(membrane_time_constants, dtype=float)


time_series_data = initialize_data(num_time_steps, num_batches, num_input_channels,
                                   start_value, end_value)

(input_weights_tensor, recurrent_weights_tensor,
 output_weights_tensor, time_series_data_tensor,
 membrane_time_constants_tensor) = convert_to_tensors(input_weights, recurrent_weights, output_weights,
                                                     time_series_data, membrane_time_constants)


# -------------------------------------------------------------------
# FORWARD PASS
# -------------------------------------------------------------------

spiking_module = tf.load_op_library("./spiking_network.so")

start = time.time()
(resulting_voltages_tensor,
 resulting_activations_tensor) = spiking_module.forward_pass(input_weights_tensor,
                                                             recurrent_weights_tensor,
                                                             membrane_time_constants_tensor,
                                                             time_series_data_tensor,
                                                             threshold_voltage=threshold_voltage,
                                                             delta_t=dt,
                                                             debug_mode=debug_mode)

cuda_forward_duration = time.time() - start

resulting_voltages = resulting_voltages_tensor.numpy().astype(np.single)
resulting_activations = resulting_activations_tensor.numpy().astype(np.single)

# Python implementation to check against:

start = time.time()

expected_voltages, expected_activations = python_forward_pass(input_weights, recurrent_weights, membrane_time_constants,
                                                              time_series_data, threshold_voltage, dt)

python_forward_duration = time.time() - start

# -------------------------------------------------------------------
# BACKWARD PASS
# -------------------------------------------------------------------
expected_output = (time_series_data[:, :, 0] + time_series_data[:, :, 1]).reshape(num_time_steps,
                                                                                  num_batches,
                                                                                  num_output_channels)

network_output = np.dot(resulting_voltages, output_weights.T)

dE_dy = -2 / num_time_steps * (expected_output - network_output)
partial_dy_dv = output_weights
partial_dE_dv = np.dot(dE_dy, partial_dy_dv).astype(np.single)
partial_dE_dv_tensor = tf.convert_to_tensor(partial_dE_dv, dtype=float)

start = time.time()
(expected_dE_dW_in,
 expected_dE_dW_rec,
 expected_dE_dmembrane_time_constants) = python_backward_pass(time_series_data,
                                                              resulting_voltages, resulting_activations,
                                                              partial_dE_dv,
                                                              recurrent_weights, membrane_time_constants,
                                                              threshold_voltage, dt,
                                                              dampening_factor=gradient_scaling_factor)

python_backward_duration = time.time() - start

start = time.time()

(resulting_dE_dW_in_tensor,
 resulting_dE_dW_rec_tensor,
 resulting_dE_dmembrane_time_constants_tensor) = spiking_module.backward_pass(partial_dE_dv_tensor,
                                                                              recurrent_weights_tensor,
                                                                              membrane_time_constants_tensor,
                                                                              time_series_data_tensor,
                                                                              resulting_voltages_tensor,
                                                                              resulting_activations_tensor,
                                                                              threshold_voltage=threshold_voltage,
                                                                              delta_t=dt,
                                                                              gradient_scaling_factor=gradient_scaling_factor,
                                                                              debug_mode=debug_mode)

cuda_backward_duration = time.time() - start

resulting_dE_dW_in = resulting_dE_dW_in_tensor.numpy()
resulting_dE_dW_rec = resulting_dE_dW_rec_tensor.numpy()
resulting_dE_dmembrane_time_constants = resulting_dE_dmembrane_time_constants_tensor.numpy()

# -------------------------------------------------------------------
# VALIDATING THE RESULTS
# -------------------------------------------------------------------

(cuda_voltages_ok, cuda_activations_ok,
 python_voltages_ok, python_activations_ok,
 voltages_match, activations_match) = verify_forward_pass(expected_voltages, expected_activations,
                                                          resulting_voltages, resulting_activations)


(input_weights_match,
 recurrent_weights_match,
 membrane_time_constants_match) = verify_backward_pass(expected_dE_dW_in, expected_dE_dW_rec,
                                                       expected_dE_dmembrane_time_constants,
                                                       resulting_dE_dW_in, resulting_dE_dW_rec,
                                                       resulting_dE_dmembrane_time_constants)
if verbose:
    print("Forward Pass Results: ")

    #if cuda_voltages_ok:

    #    print("Membrane voltages:")
    #    print(resulting_voltages[:, 0])

    #    print("\n--------------------------\n")

    #    print("Neuron activations")
    #    print(resulting_activations[:, 0])

    #   print("Activated voltages")

    #    print("\n--------------------------\n")

    #    print(resulting_voltages[:, 0][resulting_activations[:, 0].astype(bool)])

    #else:
    #    print('Voltages do not match within batches for each neuron')

    print("\nCUDA Implementation:")
    print(f"Are all voltages close to equal between batches? {cuda_voltages_ok}")
    print(f"Are all activations close to equal between batches? {cuda_activations_ok}")

    print("\nPython Implementation")
    print(f"Are all voltages close to equal between batches? {python_voltages_ok}")
    print(f"Are all activations close to equal between batches? {python_activations_ok}")

    print("\nComparing implementations:")
    print(f"Do the membrane voltages match? {voltages_match}")
    print(f"Do the neuron activations match between Python and CUDA? {activations_match}")

    if not voltages_match:
        print_voltage_discrepancies(expected_voltages, resulting_voltages)

    print("\nBackward Pass Results: ")
    print(f"Do the input weights match between Python and CUDA: {input_weights_match}")
    print(f"Do the recurrent weights match between Python and CUDA: {recurrent_weights_match}")
    print(f"Do the membrane time constants match between Python and CUDA: {membrane_time_constants_match}")

    if not input_weights_match:
        print_weight_discrepancies("Input",
                                   expected_dE_dW_in,
                                   resulting_dE_dW_in)

    if not recurrent_weights_match:
        print_weight_discrepancies("Recurrent",
                                   expected_dE_dW_rec,
                                   resulting_dE_dW_rec)

    if not membrane_time_constants_match:
        print_weight_discrepancies("Membrane Time Constant",
                                   expected_dE_dmembrane_time_constants,
                                   resulting_dE_dmembrane_time_constants)

    print("\nRuntimes:")
    print("Forward Pass")
    print(f"Runtime CUDA OP: {round(cuda_forward_duration, 5)}")
    print(f"Runtime native Python: {round(python_forward_duration, 5)}")

    print("\nBackward Pass")
    print(f"Runtime CUDA OP: {round(cuda_backward_duration, 5)}")
    print(f"Runtime native Python: {round(python_backward_duration, 5)}")

    print("\nTotal Runtimes:")
    print(f"CUDA: {round(cuda_forward_duration + cuda_backward_duration, 5)}")
    print(f"Python: {round(python_forward_duration + python_backward_duration, 5)}")

# -------------------------------------------------------------------
# FINISH UP
# -------------------------------------------------------------------

if save_data:
    hyperparameters = np.array((num_time_steps,
                                num_batches,
                                num_neurons,
                                num_input_channels,
                                num_output_channels,
                                decay_factor,
                                threshold_voltage))

    save_to_pickle_files(hyperparameters,
                         input_weights, recurrent_weights, membrane_time_constants, output_weights,
                         time_series_data, resulting_voltages, resulting_activations,
                         expected_dE_dW_in, expected_dE_dW_rec,
                         resulting_dE_dW_in, resulting_dE_dW_rec)

    #with open(os.path.join(".",  "data", "gradient_data", "expected_alpha_components.p"), "wb") as pickle_file:
    #    pickle.dump(membrane_decay_components, pickle_file)

    #with open(os.path.join(".",  "data", "gradient_data", "resulting_alpha_components.p"), "wb") as pickle_file:
    #    pickle.dump(resulting_alpha_components_tensor.numpy(), pickle_file)

    #with open(os.path.join(".",  "data", "gradient_data", "total_derivatives.p"), "wb") as pickle_file:
    #    pickle.dump(total_derivatives, pickle_file)

    #with open(os.path.join(".",  "data", "gradient_data", "expected_dE_dalpha.p"), "wb") as pickle_file:
    #    pickle.dump(expected_dE_dalpha, pickle_file)

    #with open(os.path.join(".",  "data", "gradient_data", "resulting_dE_dalpha.p"), "wb") as pickle_file:
    #    pickle.dump(resulting_dE_dalpha, pickle_file)

    #with open(os.path.join(".",  "data", "gradient_data", "dE_dalpha_progress.p"), "wb") as pickle_file:
    #    pickle.dump(alpha_progress, pickle_file)

    #with open(os.path.join(".",  "data", "gradient_data", "resulting_input_gradients.p"), "wb") as pickle_file:
    #    pickle.dump(resulting_input_gradients.numpy(), pickle_file)

    #with open(os.path.join(".",  "data", "gradient_data", "resulting_recurrent_gradients.p"), "wb") as pickle_file:
    #    pickle.dump(resulting_recurrent_gradients.numpy(), pickle_file)

    #with open(os.path.join(".",  "data", "gradient_data", "partial_gradients.p"), "wb") as pickle_file:
    #    pickle.dump(partial_dE_dv, pickle_file)



