import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
import pickle
import os

from rsnn_utils.rsnn import initialize_weights, convert_to_tensors, spike_gradient


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

weights_directory = "./data/weights"

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

t_start = 0
t_end = 4 * np.pi
num_time_steps = 1000

dt = (t_end - t_start)/num_time_steps
initial_membrane_time_constant = 20 / 1000  # 20 ms

num_input_channels = 2
num_neurons = 8
num_output_channels = 2
num_batches = 1

output_time_window = 20

decay_factor = np.exp(-dt/initial_membrane_time_constant)
threshold_voltage = 1
gradient_scaling_factor = 0.3

learning_rate = 5e-3
momentum_beta = 0.9
RMS_beta = 0.999
epsilon = 1e-8
num_epochs = 1_000
regularization_lambda = 0.0


# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------


(W_in, W_rec, W_out, tau_membrane) = initialize_weights(num_neurons, num_input_channels, num_output_channels,
                                                        threshold_voltage, initial_membrane_time_constant)

time_vector = np.linspace(t_start, t_end, num_time_steps)

input_1 = np.cos(time_vector)
input_2 = np.sin(time_vector)

time_series_data = np.array((input_1,
                             input_2)).T.reshape(num_time_steps, 1, num_input_channels)
time_series_data = np.repeat(time_series_data, num_batches, axis=1)

(W_in_tensor, W_rec_tensor,
 W_out_tensor, time_series_data_tensor,
 tau_membrane_tensor) = convert_to_tensors(W_in, W_rec, W_out,
                                           time_series_data, tau_membrane)

target = np.array((input_1 < input_2,
                   input_1 > input_2)).astype(float).T

target = target.reshape(num_time_steps, 1, num_output_channels)
target = np.repeat(target, num_batches, axis=1)

# sum-to-one condition
assert np.all(np.sum(target, axis=-1) == 1)


total_loss_over_epochs = []
original_loss_over_epochs = []
regularization_loss_over_epochs = []

one_tenth = int(num_epochs / 10)

# -------------------------------------------------------------------
# LOAD IN CUDA FUNCTIONS
# -------------------------------------------------------------------

rsnn = tf.load_op_library("./spiking_network.so")

# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------

for epoch in range(1, num_epochs + 1):

    # Forward Pass
    (resulting_voltages_tensor,
     resulting_activations_tensor) = rsnn.forward_pass(W_in_tensor, W_rec_tensor,
                                                       tau_membrane_tensor,
                                                       time_series_data_tensor,
                                                       threshold_voltage=threshold_voltage,
                                                       delta_t=dt)

    resulting_voltages = resulting_voltages_tensor.numpy()
    resulting_activations = resulting_activations_tensor.numpy()

    smoothed_spikes = np.zeros_like(resulting_activations)
    for i in range(output_time_window, num_time_steps):
        smoothed_spikes[i] = np.mean(resulting_activations[i - output_time_window: i], axis=0)

    network_output = np.dot(smoothed_spikes, W_out.T)

    softmax_output = np.exp(network_output - np.max(network_output))
    softmax_output = softmax_output / np.sum(softmax_output, axis=-1, keepdims=True)

    # Calculate Loss
    current_original_loss = - 1 / num_time_steps * np.sum(target[output_time_window:] * np.log(softmax_output[output_time_window:]))
    current_regularization_loss = regularization_lambda / 2 * (np.sum(W_in ** 2) + np.sum(W_rec ** 2) + np.sum(W_out ** 2) + np.sum(tau_membrane ** 2))
    current_total_loss = round(current_original_loss + current_regularization_loss, 6)

    original_loss_over_epochs.append(current_original_loss)
    regularization_loss_over_epochs.append(current_regularization_loss)
    total_loss_over_epochs.append(current_total_loss)

    if epoch == 1 or epoch % one_tenth == 0:
        print(f"Epoch {epoch}: {current_total_loss}")

    if current_total_loss > 1e10:
        print("Gradient Descent is diverging!")
        break

    # Backward Pass

    # Calculate the partial derivatives w.r.t. E
    dE_dnetwork_output = softmax_output - target

    dE_dW_out = np.zeros_like(W_out)
    dE_dsmoothed_spikes = np.zeros_like(smoothed_spikes)

    for t in range(num_time_steps):
        dE_dW_out_component = np.dot(dE_dnetwork_output[t].T, smoothed_spikes[t])

        dE_dW_out += dE_dW_out_component

        dE_dsmoothed_spikes[t] = np.dot(dE_dnetwork_output[t], W_out)

    # Derivative w.r.t. the neuron spikes
    dE_dz = np.zeros_like(resulting_activations)

    for i in range(output_time_window, num_time_steps):
        dE_dz[i] = np.mean(dE_dsmoothed_spikes[i - output_time_window: i], axis=0)

    partial_dE_dv = dE_dz * spike_gradient(resulting_voltages, threshold_voltage)

    partial_dE_dv_tensor = tf.convert_to_tensor(partial_dE_dv, dtype=float)

    # Calculate the total derivatives w.r.t. the loss function E
    (dE_dW_in_tensor,
     dE_dW_rec_tensor,
     dE_dtau_membrane_tensor) = rsnn.backward_pass(partial_dE_dv_tensor, W_rec_tensor, tau_membrane_tensor,
                                                   time_series_data_tensor,
                                                   resulting_voltages_tensor, resulting_activations_tensor,
                                                   threshold_voltage=threshold_voltage, delta_t=dt,
                                                   gradient_scaling_factor=gradient_scaling_factor)

    # Update the parameters
    W_in -= learning_rate * (dE_dW_in_tensor.numpy() + regularization_lambda * W_in)
    W_rec -= learning_rate * (dE_dW_rec_tensor.numpy() + regularization_lambda * W_rec)
    W_out -= learning_rate * (dE_dW_out + regularization_lambda * W_out)
    # division by num_time_steps for small learning rate. Needed for numeric stability
    tau_membrane -= learning_rate / num_time_steps * (dE_dtau_membrane_tensor.numpy() +
                                                                 regularization_lambda * tau_membrane)

    # Sanity checks
    np.fill_diagonal(W_rec, -threshold_voltage)

    if np.any(tau_membrane < 0.0):
        print("Time constant smaller than 0")

    if np.any(tau_membrane > 10):
        print("Time constant larger than 10")

    tau_membrane[tau_membrane < 0.0] = 1e-3
    tau_membrane[tau_membrane > 10.0] = 10.0

    W_in_tensor = tf.convert_to_tensor(W_in, dtype=float)
    W_rec_tensor = tf.convert_to_tensor(W_rec, dtype=float)
    tau_membrane_tensor = tf.convert_to_tensor(tau_membrane, dtype=float)


print("Training completed!")

os.makedirs(weights_directory, exist_ok=True)

with open(os.path.join(weights_directory, "W_in.p"), mode='wb') as pickle_file:
    pickle.dump(W_in_tensor.numpy(), pickle_file)

with open(os.path.join(weights_directory, "W_rec.p"), mode='wb') as pickle_file:
    pickle.dump(W_rec_tensor.numpy(), pickle_file)

with open(os.path.join(weights_directory, "W_out.p"), mode='wb') as pickle_file:
    pickle.dump(W_out, pickle_file)

with open(os.path.join(weights_directory, "tau_membrane.p"), mode='wb') as pickle_file:
    pickle.dump(tau_membrane_tensor.numpy(), pickle_file)

with open(os.path.join(weights_directory, "time_series_data.p"), mode='wb') as pickle_file:
    pickle.dump(time_series_data_tensor.numpy(), pickle_file)

hyperparameters = (t_start, t_end, num_time_steps, num_epochs,
                   num_input_channels, num_neurons,
                   threshold_voltage, dt, initial_membrane_time_constant,
                   output_time_window)

with open(os.path.join(weights_directory, "hyperparameters.p"), mode='wb') as pickle_file:
    pickle.dump(hyperparameters, pickle_file)

with open(os.path.join(weights_directory, "total_loss_over_epochs.p"), mode='wb') as pickle_file:
    pickle.dump(np.array(total_loss_over_epochs), pickle_file)

with open(os.path.join(weights_directory, "original_loss_over_epochs.p"), mode='wb') as pickle_file:
    pickle.dump(np.array(original_loss_over_epochs), pickle_file)

with open(os.path.join(weights_directory, "regularization_loss_over_epochs.p"), mode='wb') as pickle_file:
    pickle.dump(np.array(regularization_loss_over_epochs), pickle_file)

with open(os.path.join(weights_directory, "target.p"), mode='wb') as pickle_file:
    pickle.dump(target, pickle_file)

print("Data has been saved")
