import numpy as np
import matplotlib.pyplot as plt

from snn_utils import initialize_weights, spike_gradient, python_forward_pass, python_backward_pass
# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

t_start = 0
t_end = 4 * np.pi

num_time_steps = 1000
num_input_channels = 2
num_neurons = 8
num_output_channels = 2
num_batches = 1

output_time_window = 20
threshold_voltage = 1
dt = (t_end - t_start)/num_time_steps
tau_membrane = 20 / 1_000

learning_rate = 5e-3
momentum_beta = 0.9
RMS_beta = 0.999
epsilon = 1e-8
num_epochs = 100  # 1_000
regularization_lambda = 0.05

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------

initial_decay_factor = np.exp(-dt/tau_membrane)
print(initial_decay_factor)
membrane_time_constants = tau_membrane * np.ones((num_neurons, 1))

time_vector = np.linspace(t_start, t_end, num_time_steps)

input_1 = np.cos(time_vector)
input_2 = np.sin(time_vector)

time_series_data = np.array((input_1,
                             input_2)).T.reshape(num_time_steps, 1, num_input_channels)
time_series_data = np.repeat(time_series_data, num_batches, axis=1)


target = np.array((input_1 < input_2,
                   input_1 > input_2)).astype(float).T

target = target.reshape(num_time_steps, 1, num_output_channels)
target = np.repeat(target, num_batches, axis=1)

assert np.all(np.sum(target, axis=-1) == 1)

W_in, W_rec, W_out = initialize_weights(num_neurons, num_input_channels, num_output_channels)


total_loss_over_epochs = []
original_loss_over_epochs = []
regularization_loss_over_epochs = []

one_tenth = int(num_epochs / 10)

# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------

for epoch in range(1, num_epochs + 1):

    membrane_decay_factors = np.exp(-dt / membrane_time_constants)

    # Forward Pass
    resulting_voltages, resulting_activations = python_forward_pass(W_in, W_rec, membrane_decay_factors,
                                                                    time_series_data, threshold_voltage)

    smoothed_spikes = np.zeros_like(resulting_activations)
    for i in range(output_time_window, num_time_steps):
        smoothed_spikes[i] = np.mean(resulting_activations[i - output_time_window: i], axis=0)

    network_output = np.dot(smoothed_spikes, W_out.T)

    softmax_output = np.exp(network_output - np.max(network_output))
    softmax_output = softmax_output / np.sum(softmax_output, axis=-1, keepdims=True)

    # Calculate Loss
    current_function_loss = - 1 / num_time_steps * np.sum(target[output_time_window:] * np.log(softmax_output[output_time_window:]))
    current_regularization_loss = regularization_lambda / 2 * (np.sum(W_in ** 2) + np.sum(W_rec ** 2) + np.sum(W_out ** 2) + np.sum(membrane_time_constants ** 2))
    current_total_loss = round(current_function_loss + current_regularization_loss, 6)

    original_loss_over_epochs.append(current_function_loss)
    regularization_loss_over_epochs.append(current_regularization_loss)
    total_loss_over_epochs.append(current_total_loss)

    if epoch == 1 or epoch % one_tenth == 0:
        print(f"Epoch {epoch}: {current_total_loss}")

    # Backward Pass

    # Calculate the partial derivatives wrt E
    dE_dnetwork_output = softmax_output - target

    dE_dW_out = np.zeros_like(W_out)
    dE_dsmoothed_spikes = np.zeros_like(smoothed_spikes)

    for t in range(num_time_steps):
        dE_dW_out_component = np.dot(dE_dnetwork_output[t].T, smoothed_spikes[t])

        dE_dW_out += dE_dW_out_component

        dE_dsmoothed_spikes[t] = np.dot(dE_dnetwork_output[t], W_out)

    dE_dz = np.zeros_like(resulting_activations)
    for i in range(output_time_window, num_time_steps):
        dE_dz[i] = np.mean(dE_dsmoothed_spikes[i - output_time_window: i], axis=0)

    partial_dE_dv = dE_dz * spike_gradient(resulting_voltages, threshold_voltage)

    # Calculate the total derivatives wrt E
    dE_dW_in, dE_dW_rec, dE_dalpha = python_backward_pass(time_series_data, resulting_voltages, resulting_activations,
                                                          partial_dE_dv, W_rec, membrane_decay_factors,
                                                          threshold_voltage, dampening_factor=0.3)

    dE_dmembrane_time_constants = dE_dalpha * dt / membrane_time_constants ** 2 * membrane_decay_factors

    # Update the parameters
    W_in -= learning_rate * (dE_dW_in + regularization_lambda * W_in)
    W_rec -= learning_rate * (dE_dW_rec + regularization_lambda * W_rec)
    W_out -= learning_rate * (dE_dW_out + regularization_lambda * W_out)
    # division by num_time_steps for small learning rate. Needed for numeric stability
    membrane_time_constants -= learning_rate / num_time_steps * (dE_dmembrane_time_constants + regularization_lambda * membrane_time_constants)

    # Sanity checks
    np.fill_diagonal(W_rec, -threshold_voltage)

    if np.any(membrane_time_constants < 0.0):
        print("Time constant smaller than 0")

    if np.any(membrane_time_constants > 10):
        print("Time constant larger than 10")

    membrane_time_constants[membrane_time_constants < 0.0] = 1e-3
    membrane_time_constants[membrane_time_constants > 10.0] = 10.0

# -------------------------------------------------------------------
# RESULTS AND VISUALISATIONS
# -------------------------------------------------------------------

fig = plt.figure(figsize=(10, 10))

fig.suptitle("Loss over epochs")

plt.plot(original_loss_over_epochs, label="Original Loss")
plt.plot(regularization_loss_over_epochs, label="Regularization Loss")
plt.plot(total_loss_over_epochs, label="Total Loss")

plt.legend()

plt.show()

membrane_decay_factors = np.exp(-dt/membrane_time_constants)
resulting_voltages, resulting_activations = python_forward_pass(W_in, W_rec, membrane_decay_factors,
                                                                time_series_data, threshold_voltage)

smoothed_spikes = np.array([np.sum(resulting_activations[np.maximum(i - output_time_window, 0) : i], axis=0)/output_time_window for i in range(0, num_time_steps)])
network_output = np.dot(smoothed_spikes, W_out.T)

softmax_output = np.exp(network_output - np.max(network_output))
softmax_output = softmax_output/np.sum(softmax_output, axis=-1, keepdims=True)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(15, 15))

ax1.set_title("Input Data")
ax1.plot(time_vector, input_1, label="Input Channel 1")
ax1.plot(time_vector, input_2, label="Input Channel 2")

ax2.set_title("Output Channel 1")
ax2.plot(time_vector, softmax_output[:, 0, 0], label="Network Output")
ax2.plot(time_vector, target[:, 0, 0], "--", label="Target")

ax3.set_title("Output Channel 2")
ax3.plot(time_vector, softmax_output[:, 0, 1], label="Network Output")
ax3.plot(time_vector, target[:, 0, 1], "--", label="Target")

ax4.pcolormesh(resulting_activations[:, 0].T, cmap="Greys")

ax1.legend(fontsize=13)
ax2.legend(fontsize=13)
ax3.legend(fontsize=13)



plt.show()
