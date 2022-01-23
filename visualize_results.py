import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from snn_utils import python_forward_pass

weights_directory = "./data/weights"

with open(os.path.join(weights_directory, "W_in.p"), mode='rb') as pickle_file:
    W_in = pickle.load(pickle_file)

with open(os.path.join(weights_directory, "W_rec.p"), mode='rb') as pickle_file:
    W_rec = pickle.load(pickle_file)

with open(os.path.join(weights_directory, "W_out.p"), mode='rb') as pickle_file:
    W_out = pickle.load(pickle_file)

with open(os.path.join(weights_directory, "tau_membrane.p"), mode='rb') as pickle_file:
    tau_membrane = pickle.load(pickle_file)

with open(os.path.join(weights_directory, "time_series_data.p"), mode='rb') as pickle_file:
    time_series_data = pickle.load(pickle_file)

with open(os.path.join(weights_directory, "hyperparameters.p"), mode='rb') as pickle_file:
    hyperparameters = pickle.load(pickle_file)

(t_start, t_end, num_time_steps, num_epochs,
 num_input_channels, num_neurons,
 threshold_voltage, dt, initial_membrane_time_constant,
 output_time_window) = hyperparameters

time_vector = np.linspace(t_start, t_end, num_time_steps)

input_1, input_2, *_ = np.split(time_series_data, num_input_channels, axis=-1)
input_1 = input_1.flatten()
input_2 = input_2.flatten()

with open(os.path.join(weights_directory, "total_loss_over_epochs.p"), mode='rb') as pickle_file:
    total_loss_over_epochs = pickle.load(pickle_file)

with open(os.path.join(weights_directory, "original_loss_over_epochs.p"), mode='rb') as pickle_file:
    original_loss_over_epochs = pickle.load(pickle_file)

with open(os.path.join(weights_directory, "regularization_loss_over_epochs.p"), mode='rb') as pickle_file:
    regularization_loss_over_epochs = pickle.load(pickle_file)

with open(os.path.join(weights_directory, "target.p"), mode='rb') as pickle_file:
    target = pickle.load(pickle_file)

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

resulting_voltages, resulting_activations = python_forward_pass(W_in, W_rec, tau_membrane,
                                                                time_series_data,
                                                                threshold_voltage, dt)

smoothed_spikes = np.zeros_like(resulting_activations)
for i in range(output_time_window, num_time_steps):
    smoothed_spikes[i] = np.mean(resulting_activations[i - output_time_window: i], axis=0)

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

fig_3 = plt.figure(figsize=(num_neurons, 7))

fig_3.suptitle(f"Membrane time constants after training for {num_epochs} epochs")

plt.bar(range(1, num_neurons + 1), tau_membrane.flatten() * 1000)
plt.axhline(y=initial_membrane_time_constant * 1000, linestyle="--", color="crimson")

plt.xlabel("Neuron", fontsize=13)
plt.ylabel("Time constants [ms]", fontsize=13)
# plt.ylim([0, 1])

plt.show()
