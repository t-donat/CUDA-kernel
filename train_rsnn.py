import os
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import tensorflow as tf

from snn_utils import initialize_weights, python_forward_pass, find_class_with_max_probability, python_backward_pass, spike_gradient, evaluate_model

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

num_neurons = 128

dt = 1/sampling_frequency
initial_membrane_time_constant = 20/1000

output_time_window = 20
threshold_voltage = 1
gradient_scaling_factor = 0.3

learning_rate = 5e-3
num_epochs = 500
one_tenth = int(num_epochs/10)

# -------------------------------------------------------------------
# LOAD IN CUDA FUNCTIONS
# -------------------------------------------------------------------

rsnn = tf.load_op_library("./spiking_network.so")

# -------------------------------------------------------------------
# LOAD IN DATA
# -------------------------------------------------------------------

with open(os.path.join("..", "BCI_Data", "Data", "dataset", "dataset", "train_data_set.p"), "rb") as pickle_file:
    train_data_set = pickle.load(pickle_file)

with open(os.path.join("..", "BCI_Data", "Data", "dataset", "dataset", "test_data_set.p"), "rb") as pickle_file:
    test_data_set = pickle.load(pickle_file)

with open(os.path.join("..", "BCI_Data", "Data", "dataset", "dataset", "hyperparameters.json"), "r") as file:
    hyperparameters = json.load(file)

sampling_frequency = hyperparameters["sampling_frequency"]
num_classes = hyperparameters["num_classes"]
num_time_steps = hyperparameters["num_time_steps"]
num_input_channels = hyperparameters["num_input_channels"]

train_samples, train_labels = train_data_set
test_samples, test_labels = test_data_set

(W_in, W_rec, W_out, tau_membrane) = initialize_weights(num_neurons, num_input_channels, num_classes,
                                                        threshold_voltage, initial_membrane_time_constant)

loss_over_epochs = []
accuracy_over_epochs = []

for current_epoch in range(1, num_epochs+1):
    for batch_data, batch_labels in zip(train_samples, train_labels):
        current_batch_size = batch_data.shape[1]

        resulting_voltages, resulting_activations = python_forward_pass(W_in, W_rec, tau_membrane,
                                                                        batch_data,
                                                                        threshold_voltage, dt)

        smoothed_spikes = np.zeros_like(resulting_activations)
        for i in range(output_time_window, num_time_steps):
            smoothed_spikes[i] = np.mean(resulting_activations[i - output_time_window: i], axis=0)

        network_output = np.dot(smoothed_spikes, W_out.T)

        softmax_output = np.exp(network_output - np.max(network_output))
        softmax_output = softmax_output / np.sum(softmax_output, axis=-1, keepdims=True)

        batch_costs = []
        batch_correct_predictions = []

        dE_dW_out = np.zeros_like(W_out)
        dE_dnetwork_output = np.zeros_like(network_output)
        dE_dsmoothed_spikes = np.zeros_like(smoothed_spikes)
        dE_dz = np.zeros_like(resulting_activations)

        for b in range(current_batch_size):
            highest_prob_class, highest_prob_time_step = find_class_with_max_probability(softmax_output[:, b])

            predicted_distribution = softmax_output[highest_prob_time_step, b]
            ground_truth_distribution = batch_labels[b]

            cost = - ground_truth_distribution * np.log(predicted_distribution)

            batch_costs.append(cost)
            batch_correct_predictions.append(ground_truth_distribution[highest_prob_class] == 1)

            dE_dnetwork_output[highest_prob_time_step, b] = predicted_distribution - ground_truth_distribution

        for t in range(num_time_steps):
                dE_dW_out_component = np.dot(dE_dnetwork_output[t].T, smoothed_spikes[t])

                dE_dW_out += dE_dW_out_component

                dE_dsmoothed_spikes[t] = np.dot(dE_dnetwork_output[t], W_out)

        for i in range(output_time_window, num_time_steps):
            dE_dz[i] = np.mean(dE_dsmoothed_spikes[i - output_time_window: i], axis=0)

        partial_dE_dv = dE_dz * spike_gradient(resulting_voltages, threshold_voltage)

        (dE_dW_in,
         dE_dW_rec,
         dE_dtau_membrane) = python_backward_pass(batch_data,
                                                  resulting_voltages, resulting_activations,
                                                  partial_dE_dv,
                                                  W_rec, tau_membrane,
                                                  threshold_voltage, dt)

        W_in -= learning_rate * dE_dW_in
        W_rec -= learning_rate * dE_dW_rec
        W_out -= learning_rate * dE_dW_out
        tau_membrane -= learning_rate/num_time_steps * dE_dtau_membrane

    loss_of_epoch = np.sum(batch_costs)/current_batch_size
    accuracy_of_epoch = np.mean(batch_correct_predictions)

    loss_over_epochs.append(loss_of_epoch)
    accuracy_over_epochs.append(accuracy_of_epoch)

    if current_epoch == 1 or current_epoch % one_tenth == 0:
        print(f"Epoch {current_epoch}: Loss: {round(loss_of_epoch, 3)}, Accuracy: {round(accuracy_of_epoch, 3)}")


test_set_accuracy = evaluate_model(W_in, W_rec, W_out, tau_membrane,
                   output_time_window, threshold_voltage, dt, num_time_steps,
                   test_samples, test_labels)

fig1 = plt.figure(1, figsize=(10, 10))
fig1.suptitle(f"Accuracy over {num_epochs} epochs. Final accuracy: {round(test_set_accuracy * 100, 2)}%")
plt.plot(accuracy_over_epochs)
plt.show()

fig2 = plt.figure(2, figsize=(10, 10))
fig2.suptitle(f"Loss over {num_epochs} epochs")
plt.plot(loss_over_epochs)
plt.show()

fig3 = plt.figure(3, figsize=(10, 10))
fig3.suptitle(f"Neuron activations of batch 0")
plt.pcolormesh(resulting_activations[:, 0].T, cmap="Greys")
plt.show()

with open(os.path.join("..", "BCI_Data", "Data", "dataset", "dataset", "W_in.p"), "wb") as pickle_file:
    pickle.dump(W_in, pickle_file)

with open(os.path.join("..", "BCI_Data", "Data", "dataset", "dataset", "W_rec.p"), "wb") as pickle_file:
    pickle.dump(W_rec, pickle_file)

with open(os.path.join("..", "BCI_Data", "Data", "dataset", "dataset", "W_out.p"), "wb") as pickle_file:
    pickle.dump(W_out, pickle_file)

with open(os.path.join("..", "BCI_Data", "Data", "dataset", "dataset", "tau_membrane.p"), "wb") as pickle_file:
    pickle.dump(tau_membrane, pickle_file)

