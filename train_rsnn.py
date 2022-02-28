import os
import numpy as np
import pickle
import json
import argparse
import time
# import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from rsnn_utils.rsnn import initialize_weights, calculate_spike_gradient, convert_batch_to_tensors
from rsnn_utils.data import evaluate_model, find_indices_of_max_probabilities

physical_devices = tf.config.experimental.list_physical_devices('GPU')

for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# -------------------------------------------------------------------
# LOAD IN DATA
# -------------------------------------------------------------------

parser = argparse.ArgumentParser("Train the RSNN")
parser.add_argument("-i", "--input_dir",
                    dest="input_directory",
                    help="Directory containing the train and test data")

parser.add_argument("-o", "--output_dir",
                    dest="output_directory",
                    help="Directory to save the results to")

parser.add_argument("-j", "--job_id",
                    dest="job_id",
                    help="ID of the Slurm Job calling this script",
                    default="0000")

parser.add_argument("-e", "--num_epochs",
                    dest="num_epochs",
                    help="Number of epochs to train for")

parser.add_argument("-n", "--num_neurons",
                    dest="num_neurons",
                    help="Number of spiking neurons in the RSNN")

parser.add_argument("--val",
                    help="Whether to expect and use a validation set",
                    action="store_true",
                    default=False,
                    dest="use_validation_set")

arguments = parser.parse_args()
input_directory = arguments.input_directory
output_directory = arguments.output_directory
job_id = arguments.job_id
num_epochs = int(arguments.num_epochs)
num_neurons = int(arguments.num_neurons)
use_validation_set = arguments.use_validation_set

with open(os.path.join(input_directory, "train_data_set.p"), "rb") as pickle_file:
    train_data_set = pickle.load(pickle_file)

if use_validation_set:
    with open(os.path.join(input_directory, "validation_data_set.p"), "rb") as pickle_file:
        validation_data_set = pickle.load(pickle_file)

    validation_samples, validation_labels = validation_data_set
    del validation_data_set

with open(os.path.join(input_directory, "hyperparameters.json"), "r") as file:
    hyperparameters = json.load(file)

sampling_frequency = hyperparameters["sampling_frequency"]
num_classes = hyperparameters["num_classes"]
num_time_steps = hyperparameters["num_time_steps"]
num_input_channels = hyperparameters["num_input_channels"]

train_samples, train_labels = train_data_set
del train_data_set

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

# num_neurons = 128

dt = 1 / sampling_frequency
initial_membrane_time_constant = 100 / 1000

output_time_window = 100
threshold_voltage = 2
expected_firing_rate = 0.1

gradient_scaling_factor = 0.3

learning_rate = 0.001
weight_decay_rate = 1e-4
firing_rate_lambda = 10_000
# num_epochs = 1

network_dropout_rate = 0.0
input_dropout_rate = 0.0

hyperparameters["num_neurons"] = num_neurons
hyperparameters["num_epochs"] = num_epochs

hyperparameters["dt"] = dt
hyperparameters["initial_membrane_time_constant"] = initial_membrane_time_constant

hyperparameters["output_time_window"] = output_time_window
hyperparameters["threshold_voltage"] = threshold_voltage
hyperparameters["expected_firing_rate"] = expected_firing_rate

hyperparameters["gradient_scaling_factor"] = gradient_scaling_factor

hyperparameters["learning_rate"] = learning_rate
hyperparameters["weight_decay_rate"] = weight_decay_rate
hyperparameters["firing_rate_lambda"] = firing_rate_lambda

hyperparameters["network_dropout_rate"] = network_dropout_rate
hyperparameters["input_dropout_rate"] = input_dropout_rate

# -------------------------------------------------------------------
# LOAD IN CUDA FUNCTIONS
# -------------------------------------------------------------------
shared_lib_path = os.path.join(os.path.dirname(__file__), "spiking_network.so")

if not os.path.exists(shared_lib_path):
    raise FileNotFoundError(f"Could not find shared library at expected path: {shared_lib_path}")
rsnn = tf.load_op_library(shared_lib_path)

(W_in_init, W_rec_init,
 W_out_init, tau_membrane_init) = initialize_weights(num_neurons, num_input_channels, num_classes,
                                                     threshold_voltage, initial_membrane_time_constant)

with tf.device("/cpu"):
    W_in = tf.Variable(W_in_init, dtype=tf.float32)
    W_rec = tf.Variable(W_rec_init, dtype=tf.float32)
    W_out = tf.Variable(W_out_init, dtype=tf.float32)
    tau_membrane = tf.Variable(tau_membrane_init, dtype=tf.float32)

optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay_rate)

# variables = [W_in, W_rec, W_out, tau_membrane]
variables = [W_in, W_rec, W_out]

train_samples_as_tensors, train_batch_sizes = convert_batch_to_tensors(train_samples)
del train_samples

debug_interval = int(np.ceil(num_epochs / 10))

train_loss_over_epochs = []
train_accuracy_over_epochs = []

if use_validation_set:
    validation_loss_over_epochs = []
    validation_accuracy_over_epochs = []

firing_rates_over_epochs = []
time_constants_over_epochs = []
time_constant_derivatives_over_epochs = []

print(f"Starting training for {num_epochs} epochs")

start = time.time()

for current_epoch in range(1, num_epochs + 1):

    loss_in_epoch = []
    accuracy_in_epoch = []

    for batch_data, batch_labels in zip(train_samples_as_tensors, train_labels):
        current_batch_size = batch_data.shape[1]

        # Input Dropout
        will_not_be_dropped = tf.cast(tf.random.uniform(shape=batch_data.get_shape()) > input_dropout_rate,
                                      dtype=tf.float32)
        dropout_corrected_batch_data = (batch_data * will_not_be_dropped) / (1 - input_dropout_rate)

        # Forward Pass
        (resulting_voltages,
         resulting_activations) = rsnn.forward_pass(W_in, W_rec, tau_membrane,
                                                    dropout_corrected_batch_data,
                                                    threshold_voltage=threshold_voltage,
                                                    delta_t=dt)
        # Network Dropout
        will_not_be_dropped = tf.cast(tf.random.uniform(shape=resulting_activations.get_shape()) > network_dropout_rate,
                                      dtype=tf.float32)

        dropout_corrected_activations = (resulting_activations * will_not_be_dropped) / (1 - network_dropout_rate)

        # Continue with Forward Pass
        smoothed_spikes = tf.stack([tf.math.reduce_mean(dropout_corrected_activations[i-output_time_window: i], axis=0)
                                    if i >= output_time_window else tf.zeros(shape=[current_batch_size, num_neurons])
                                    for i in range(num_time_steps)])

        network_output = tf.linalg.matmul(smoothed_spikes, W_out, transpose_b=True)

        softmax_output = tf.math.exp(network_output - np.max(network_output))
        softmax_output = softmax_output / tf.math.reduce_sum(softmax_output, axis=-1, keepdims=True)

        # Evaluation
        indices_with_highest_probability = find_indices_of_max_probabilities(softmax_output)
        time_step_with_highest_prob_per_sample = indices_with_highest_probability[:, :2]
        predicted_classes = indices_with_highest_probability[:, 2]

        # LOSS
        # only need the time and batch index to calculate loss
        predicted_distribution = tf.gather_nd(softmax_output, time_step_with_highest_prob_per_sample)
        # 'batch_labels' contains the one hot encoded ground truth
        ground_truth_distribution = batch_labels
        original_batch_cost = tf.math.reduce_mean(
            - ground_truth_distribution * tf.math.log(predicted_distribution)).numpy()

        actual_firing_rates = tf.math.reduce_mean(dropout_corrected_activations, axis=[0, 1])
        # MSE between actual and expected firing rates
        firing_rate_loss = firing_rate_lambda * tf.math.reduce_mean(
            (actual_firing_rates - expected_firing_rate) ** 2).numpy()
        batch_cost = original_batch_cost + firing_rate_loss

        loss_in_epoch.append(batch_cost)
        firing_rates_over_epochs.append(actual_firing_rates)

        # ACCURACY
        ground_truth_classes = tf.where(ground_truth_distribution)[:, 1]
        batch_accuracy = tf.reduce_mean(tf.cast(predicted_classes == ground_truth_classes, tf.float32)).numpy()

        # Backward Pass
        dE_dnetwork_output_values = predicted_distribution - ground_truth_distribution
        dE_dnetwork_output = tf.scatter_nd(time_step_with_highest_prob_per_sample,
                                           dE_dnetwork_output_values,
                                           network_output.get_shape())

        accuracy_in_epoch.append(batch_accuracy)

        # both matrix multiplications are calculated in batches (for each time step)
        dE_dW_out_components = tf.matmul(dE_dnetwork_output, smoothed_spikes, transpose_a=True)
        dE_dW_out = tf.math.reduce_sum(dE_dW_out_components, axis=0)
        dE_dsmoothed_spikes = tf.matmul(dE_dnetwork_output, W_out)

        original_dE_dz = tf.stack([tf.reduce_mean(dE_dsmoothed_spikes[i - output_time_window: i], axis=0)
                                   if i >= output_time_window else tf.zeros(shape=[current_batch_size, num_neurons])
                                   for i in range(num_time_steps)])

        firing_rate_dE_dz_values = firing_rate_lambda / num_neurons * (actual_firing_rates - expected_firing_rate)
        dE_dactual_firing_rate = tf.ones(shape=[num_time_steps,
                                                current_batch_size,
                                                num_neurons]) * firing_rate_dE_dz_values

        dactual_firing_rate_dz = 1/(num_time_steps * current_batch_size)

        firing_rate_dE_dz = dE_dactual_firing_rate * dactual_firing_rate_dz
        dE_dz = original_dE_dz + firing_rate_dE_dz

        partial_dE_dv = dE_dz * calculate_spike_gradient(resulting_voltages, threshold_voltage)

        (dE_dW_in,
         dE_dW_rec,
         dE_dtau_membrane) = rsnn.backward_pass(partial_dE_dv, W_rec, tau_membrane,
                                                dropout_corrected_batch_data,
                                                resulting_voltages, dropout_corrected_activations,
                                                threshold_voltage=threshold_voltage, delta_t=dt,
                                                gradient_scaling_factor=gradient_scaling_factor)

        # gradients = [dE_dW_in, dE_dW_rec, dE_dW_out, dE_dtau_membrane]
        gradients = [dE_dW_in, dE_dW_rec, dE_dW_out]

        optimizer.apply_gradients(zip(gradients, variables))

        time_constants_over_epochs.append(tau_membrane.numpy().flatten())
        time_constant_derivatives_over_epochs.append(dE_dtau_membrane.numpy().flatten())

    loss_of_epoch = tf.math.reduce_mean(loss_in_epoch).numpy()
    accuracy_of_epoch = tf.math.reduce_mean(accuracy_in_epoch).numpy()

    train_loss_over_epochs.append(loss_of_epoch)
    train_accuracy_over_epochs.append(accuracy_of_epoch)

    if use_validation_set:
        validation_set_accuracy,  validation_set_loss = evaluate_model(W_in, W_rec, W_out, tau_membrane,
                                                                       output_time_window, threshold_voltage,
                                                                       dt, num_time_steps,
                                                                       validation_samples, validation_labels,
                                                                       rsnn.forward_pass)

        validation_loss_over_epochs.append(validation_labels)
        validation_accuracy_over_epochs.append(validation_set_accuracy)

    if current_epoch == 1 or current_epoch % debug_interval == 0:

        rounded_train_loss = round(float(loss_of_epoch), 3)
        rounded_train_accuracy = round(float(accuracy_of_epoch * 100), 2)
        print(f"Epoch {current_epoch}:")
        print(f"Train: Loss: {rounded_train_loss}, Accuracy: {rounded_train_accuracy}%")

        if use_validation_set:
            rounded_validation_loss = round(float(validation_set_loss), 3)
            rounded_validation_accuracy = round(float(validation_set_accuracy * 100), 2)
            print(f"Validation: Loss: {rounded_validation_loss}, Accuracy: {rounded_validation_accuracy}%")

training_duration = time.time() - start

print("Training completed")

del train_samples_as_tensors
del train_labels

if use_validation_set:
    del validation_samples
    del validation_labels

with open(os.path.join(input_directory, "test_data_set.p"), "rb") as pickle_file:
    test_data_set = pickle.load(pickle_file)

test_samples, test_labels = test_data_set
del test_data_set

test_samples_as_tensors, test_batch_sizes = convert_batch_to_tensors(test_samples)
del test_samples

test_set_accuracy, test_set_loss = evaluate_model(W_in, W_rec, W_out, tau_membrane,
                                                  output_time_window, threshold_voltage, dt, num_time_steps,
                                                  test_samples_as_tensors, test_labels,
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


def convert_seconds_to_dhms(duration_in_seconds):
    # remove decimal points
    duration_in_seconds = int(duration_in_seconds)

    seconds = duration_in_seconds % 60
    duration_in_minutes = duration_in_seconds // 60

    minutes = duration_in_minutes % 60
    duration_in_hours = duration_in_minutes // 60

    hours = duration_in_hours % 24
    days = duration_in_hours // 24

    return f"{days}-{hours}:{minutes}:{seconds}"


last_train_accuracy = round(train_accuracy_over_epochs[-1] * 100, 2)
highest_train_accuracy = round(np.max(train_accuracy_over_epochs) * 100, 2)
last_test_accuracy = round(test_set_accuracy * 100, 2)
duration_time_string = convert_seconds_to_dhms(training_duration)

info_content = "\n".join([f"Training Results:",
                          f"Train accuracy: {last_train_accuracy}%",
                          f"Max. train accuracy: {highest_train_accuracy}%",
                          f"Test accuracy: {last_test_accuracy}%",
                          f"Test loss: {test_set_loss}",
                          f"Train duration: {duration_time_string}"])

hyperparameter_info = "\n".join([f"{key}: {value}" for key, value in hyperparameters.items()
                                 if key not in ["min_normalization_value", "max_normalization_value"]])

extra_info = "\n".join([f"Train batch sizes: {train_batch_sizes}",
                        f"Test batch sizes: {test_batch_sizes}",
                        f"Min normalization value: {hyperparameters['min_normalization_value']}",
                        f"Max normalization value: {hyperparameters['max_normalization_value']}"])

output_directory = os.path.join(output_directory,
                                f"Training_Run_{job_id}")

os.makedirs(output_directory, exist_ok=True)

with open(os.path.join(output_directory, "W_in.p"), "wb") as pickle_file:
    pickle.dump(W_in, pickle_file)

with open(os.path.join(output_directory, "W_rec.p"), "wb") as pickle_file:
    pickle.dump(W_rec, pickle_file)

with open(os.path.join(output_directory, "W_out.p"), "wb") as pickle_file:
    pickle.dump(W_out, pickle_file)

with open(os.path.join(output_directory, "tau_membrane.p"), "wb") as pickle_file:
    pickle.dump(tau_membrane, pickle_file)

with open(os.path.join(output_directory, "train_accuracy_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(train_accuracy_over_epochs, pickle_file)

with open(os.path.join(output_directory, "train_loss_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(train_loss_over_epochs, pickle_file)

if use_validation_set:
    with open(os.path.join(output_directory, "validation_loss_over_epochs.p"), "wb") as pickle_file:
        pickle.dump(validation_loss_over_epochs, pickle_file)

    with open(os.path.join(output_directory, "validation_accuracy_over_epochs.p"), "wb") as pickle_file:
        pickle.dump(validation_accuracy_over_epochs, pickle_file)

with open(os.path.join(output_directory, "firing_rates_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(firing_rates_over_epochs, pickle_file)

with open(os.path.join(output_directory, "time_constants_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(np.array(time_constants_over_epochs), pickle_file)

with open(os.path.join(output_directory, "time_constant_derivatives_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(np.array(time_constant_derivatives_over_epochs), pickle_file)

with open(os.path.join(output_directory, "info.txt"), "w") as txt_file:
    txt_file.write(info_content)
    txt_file.write("\n\nHyperparameters:\n")
    txt_file.write(hyperparameter_info)
    txt_file.write("\n\nExtra Info:\n")
    txt_file.write(extra_info)
