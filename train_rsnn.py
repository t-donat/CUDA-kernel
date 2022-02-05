import os
import numpy as np
import pickle
import json
# import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from rsnn_utils.rsnn import initialize_weights, calculate_spike_gradient, convert_to_tensors, convert_batch_to_tensors
from rsnn_utils.data import  evaluate_model, find_indices_of_max_probabilities

# -------------------------------------------------------------------
# LOAD IN DATA
# -------------------------------------------------------------------

target_dir = os.path.join("..", "Data")

with open(os.path.join(target_dir, "train_data_set.p"), "rb") as pickle_file:
    train_data_set = pickle.load(pickle_file)

with open(os.path.join(target_dir, "test_data_set.p"), "rb") as pickle_file:
    test_data_set = pickle.load(pickle_file)

with open(os.path.join(target_dir, "hyperparameters.json"), "r") as file:
    hyperparameters = json.load(file)

sampling_frequency = hyperparameters["sampling_frequency"]
num_classes = hyperparameters["num_classes"]
num_time_steps = hyperparameters["num_time_steps"]
num_input_channels = hyperparameters["num_input_channels"]

train_samples, train_labels = train_data_set
test_samples, test_labels = test_data_set

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

num_neurons = 128

dt = 1/sampling_frequency
initial_membrane_time_constant = 50/1000

output_time_window = 50
threshold_voltage = 1
gradient_scaling_factor = 0.3

learning_rate = 1e-3
weight_decay_rate = 1e-3
num_epochs = 100

# -------------------------------------------------------------------
# LOAD IN CUDA FUNCTIONS
# -------------------------------------------------------------------

rsnn = tf.load_op_library("./spiking_network.so")

(W_in_init, W_rec_init,
 W_out_init, tau_membrane_init) = initialize_weights(num_neurons, num_input_channels, num_classes,
                                                     threshold_voltage, initial_membrane_time_constant)

with tf.device("/cpu"):
    W_in = tf.Variable(W_in_init, dtype=tf.float32)
    W_rec = tf.Variable(W_rec_init, dtype=tf.float32)
    W_out = tf.Variable(W_out_init, dtype=tf.float32)
    tau_membrane = tf.Variable(tau_membrane_init, dtype=tf.float32)

optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay_rate)

variables = [W_in, W_rec, W_out, tau_membrane]

train_samples_as_tensors = convert_batch_to_tensors(train_samples)
test_samples_as_tensors = convert_batch_to_tensors(test_samples)

one_tenth = int(num_epochs/10)

loss_over_epochs = []
accuracy_over_epochs = []

print(f"Starting training for {num_epochs} epochs")

for current_epoch in range(1, num_epochs+1):
    for batch_data, batch_labels in zip(train_samples_as_tensors, train_labels):

        current_batch_size = batch_data.shape[1]

        (resulting_voltages,
         resulting_activations) = rsnn.forward_pass(W_in, W_rec, tau_membrane,
                                                    batch_data,
                                                    threshold_voltage=threshold_voltage,
                                                    delta_t=dt)
        smoothed_spikes = tf.stack([tf.math.reduce_mean(resulting_activations[i - output_time_window: i], axis=0)
                                    if i >= output_time_window else tf.zeros(shape=[current_batch_size, num_neurons])
                                    for i in range(num_time_steps)])

        network_output = tf.linalg.matmul(smoothed_spikes, W_out, transpose_b=True)

        softmax_output = tf.math.exp(network_output - np.max(network_output))
        softmax_output = softmax_output / tf.math.reduce_sum(softmax_output, axis=-1, keepdims=True)

        indices_with_highest_probability = find_indices_of_max_probabilities(softmax_output)
        time_step_with_highest_prob_per_sample = indices_with_highest_probability[:, :2]
        predicted_classes = indices_with_highest_probability[:, 2]

        # LOSS
        # only need the time and batch index to calculate loss
        predicted_distribution = tf.gather_nd(softmax_output, time_step_with_highest_prob_per_sample)
        # 'batch_labels' contains the one hot encoded ground truth
        ground_truth_distribution = batch_labels
        batch_cost = (tf.reduce_sum(- ground_truth_distribution * tf.math.log(predicted_distribution)) /
                      current_batch_size).numpy()

        # ACCURACY
        ground_truth_classes = tf.where(ground_truth_distribution)[:, 1]
        batch_accuracy = tf.reduce_mean(tf.cast(predicted_classes == ground_truth_classes, tf.float32)).numpy()

        dE_dnetwork_output_values = predicted_distribution - ground_truth_distribution
        dE_dnetwork_output = tf.scatter_nd(time_step_with_highest_prob_per_sample,
                                           dE_dnetwork_output_values,
                                           network_output.get_shape())

        # both matrix multiplications are calculated in batches (for each time step)
        dE_dW_out_components = tf.matmul(dE_dnetwork_output, smoothed_spikes, transpose_a=True)
        dE_dW_out = tf.math.reduce_sum(dE_dW_out_components, axis=0)
        dE_dsmoothed_spikes = tf.matmul(dE_dnetwork_output, W_out)

        dE_dz = tf.stack([tf.reduce_mean(dE_dsmoothed_spikes[i - output_time_window: i], axis=0)
                          if i >= output_time_window else tf.zeros(shape=[current_batch_size, num_neurons])
                          for i in range(num_time_steps)])

        partial_dE_dv = dE_dz * calculate_spike_gradient(resulting_voltages, threshold_voltage)

        (dE_dW_in,
         dE_dW_rec,
         dE_dtau_membrane) = rsnn.backward_pass(partial_dE_dv, W_rec, tau_membrane,
                                                batch_data,
                                                resulting_voltages, resulting_activations,
                                                threshold_voltage=threshold_voltage, delta_t=dt,
                                                gradient_scaling_factor=gradient_scaling_factor)

        gradients = [dE_dW_in, dE_dW_rec, dE_dW_out, dE_dtau_membrane]

        optimizer.apply_gradients(zip(gradients, variables))

        # W_in -= learning_rate * dE_dW_in
        # W_rec -= learning_rate * dE_dW_rec
        # W_out -= learning_rate * dE_dW_out
        # tau_membrane -= learning_rate/num_time_steps * dE_dtau_membrane

    loss_of_epoch = tf.math.reduce_mean(batch_cost).numpy()
    accuracy_of_epoch = tf.math.reduce_mean(batch_accuracy).numpy()

    loss_over_epochs.append(loss_of_epoch)
    accuracy_over_epochs.append(accuracy_of_epoch)

    if current_epoch == 1 or current_epoch % one_tenth == 0:
        rounded_loss = round(float(loss_of_epoch), 3)
        rounded_accuracy = round(float(accuracy_of_epoch * 100), 2)
        print(f"Epoch {current_epoch}: Loss: {rounded_loss}, Accuracy: {rounded_accuracy}%")


print("Training completed")

test_set_accuracy, test_set_loss = evaluate_model(W_in, W_rec, W_out, tau_membrane,
                                                  output_time_window, threshold_voltage, dt, num_time_steps,
                                                  test_samples, test_labels,
                                                  rsnn.forward_pass)

print("RESULTS:")
print(f"Accuracy: {round(test_set_accuracy * 100, 3)}%, Loss: {round(test_set_loss, 3)}")

# fig1 = plt.figure(1, figsize=(10, 10))
# fig1.suptitle(f"Accuracy over {num_epochs} epochs. Final accuracy: {round(test_set_accuracy * 100, 2)}%")
# plt.plot(accuracy_over_epochs)
# plt.show()

# fig2 = plt.figure(2, figsize=(10, 10))
# fig2.suptitle(f"Loss over {num_epochs} epochs")
# plt.plot(loss_over_epochs)
# plt.show()

# fig3 = plt.figure(3, figsize=(10, 10))
# fig3.suptitle(f"Neuron activations of batch 0")
# plt.pcolormesh(resulting_activations[:, 0].T, cmap="Greys")
# plt.show()

with open(os.path.join(target_dir, "W_in.p"), "wb") as pickle_file:
    pickle.dump(W_in, pickle_file)

with open(os.path.join(target_dir, "W_rec.p"), "wb") as pickle_file:
    pickle.dump(W_rec, pickle_file)

with open(os.path.join(target_dir, "W_out.p"), "wb") as pickle_file:
    pickle.dump(W_out, pickle_file)

with open(os.path.join(target_dir, "tau_membrane.p"), "wb") as pickle_file:
    pickle.dump(tau_membrane, pickle_file)

with open(os.path.join(target_dir, "accuracy_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(accuracy_over_epochs, pickle_file)

with open(os.path.join(target_dir, "loss_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(loss_over_epochs, pickle_file)

with open(os.path.join(target_dir, "test_set_results.p"), "wb") as pickle_file:
    pickle.dump((test_set_accuracy, test_set_loss), pickle_file)
