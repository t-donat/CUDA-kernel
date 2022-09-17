import os
import numpy as np
import pickle
import argparse

import tensorflow as tf

from rsnn_utils.rsnn import convert_batch_to_tensors
from rsnn_utils.data import find_indices_of_max_probabilities, create_confusion_matrix

parser = argparse.ArgumentParser("Train the RSNN")
parser.add_argument("-d", "--data_dir",
                    dest="data_directory",
                    help="Directory containing the test data set")

parser.add_argument("-i", "--input_dir",
                    dest="input_directory",
                    help="Path to the pickle file containing the batch of trials of interest")

parser.add_argument("-t", "--trial_nr",
                    dest="trial_nr",
                    help="Trial for which the confusion matrix should be calculated")

parser.add_argument("-o", "--output_dir",
                    dest="output_directory",
                    help="Directory to save the results to")

arguments = parser.parse_args()

data_directory = arguments.data_directory
input_directory = arguments.input_directory
output_directory = arguments.output_directory

trial_nr = int(arguments.trial_nr)

print("[INFO] Starting model evaluation")

print("[INFO] Loading trial data")

# TRIAL DATA
with open(input_directory, "rb") as pickle_file:
    whole_trial_batch = pickle.load(pickle_file)

trial_of_interest = whole_trial_batch[f"trial_{trial_nr}"]

W_in = trial_of_interest["validation"]["best_performing_parameters"]["W_in"]
W_rec = trial_of_interest["validation"]["best_performing_parameters"]["W_rec"]
W_out = trial_of_interest["validation"]["best_performing_parameters"]["W_out"]
tau_membrane = trial_of_interest["validation"]["best_performing_parameters"]["tau_membrane"]

num_neurons = trial_of_interest["hyperparameters"]["num_neurons"]
num_time_steps = trial_of_interest["hyperparameters"]["num_time_steps"]

output_time_window = trial_of_interest["hyperparameters"]["output_time_window"]
dt = trial_of_interest["hyperparameters"]["dt"]
threshold_voltage = trial_of_interest["hyperparameters"]["threshold_voltage"]

print("[INFO] Loading test data set")

# TEST DATA SET
# with open(os.path.join(data_directory, "test_data_set.p"), "rb") as pickle_file:
with open(os.path.join(data_directory, "validation_data_set.p"), "rb") as pickle_file:
    test_data_set = pickle.load(pickle_file)

test_samples, test_labels = test_data_set
del test_data_set

test_samples_as_tensors, test_batch_sizes = convert_batch_to_tensors(test_samples)
del test_samples

print("[INFO] Loading CUDA .so")

# LOAD IN CUDA FUNCTIONS
shared_lib_path = os.path.join(os.path.dirname(__file__), "spiking_network.so")

if not os.path.exists(shared_lib_path):
    raise FileNotFoundError(f"Could not find shared library at expected path: {shared_lib_path}")
rsnn = tf.load_op_library(shared_lib_path)

print("[INFO] Starting inference")

# EVALUATION
ground_truth_labels = []
predicted_labels = []

for batch_data, batch_labels in zip(test_samples_as_tensors, test_labels):
    current_batch_size = batch_data.shape[1]

    resulting_voltages, resulting_activations = rsnn.forward_pass(W_in, W_rec, tau_membrane,
                                                                  batch_data,
                                                                  threshold_voltage=threshold_voltage,
                                                                  delta_t=dt)

    smoothed_spikes = tf.stack([tf.math.reduce_mean(resulting_activations[i - output_time_window: i], axis=0)
                                if i >= output_time_window else tf.zeros(shape=[current_batch_size, num_neurons])
                                for i in range(1, num_time_steps + 1)])

    network_output = tf.linalg.matmul(smoothed_spikes, W_out, transpose_b=True)

    softmax_output = tf.math.exp(network_output - np.max(network_output))
    softmax_output = softmax_output / tf.math.reduce_sum(softmax_output, axis=-1, keepdims=True)

    indices_with_highest_probability = find_indices_of_max_probabilities(softmax_output)
    time_step_with_highest_prob_per_sample = indices_with_highest_probability[:, :2]
    predicted_classes = indices_with_highest_probability[:, 2]

    # current_batch_size = batch_data.shape[1]
    #
    # resulting_voltages, resulting_activations = rsnn.forward_pass(W_in, W_rec, tau_membrane,
    #                                                               batch_data,
    #                                                               threshold_voltage=threshold_voltage,
    #                                                               delta_t=dt)
    #
    # smoothed_spikes = tf.stack([tf.math.reduce_mean(resulting_activations[i - output_time_window: i], axis=0)
    #                             if i >= output_time_window else tf.zeros(shape=[current_batch_size, num_neurons])
    #                             for i in range(1, num_time_steps + 1)])
    #
    # network_output = tf.linalg.matmul(smoothed_spikes, W_out, transpose_b=True)
    #
    # softmax_output = tf.math.exp(network_output - np.max(network_output))
    # softmax_output = softmax_output / tf.math.reduce_sum(softmax_output, axis=-1, keepdims=True)
    #
    # indices_with_highest_probability = find_indices_of_max_probabilities(softmax_output)
    # time_step_with_highest_prob_per_sample = indices_with_highest_probability[:, :2]
    # predicted_classes = indices_with_highest_probability[:, 2]
    #
    # ground_truth_labels.extend(np.argmax(batch_labels, axis=1))
    # predicted_labels.extend(predicted_classes.numpy())

    ground_truth_labels.append(batch_labels)
    predicted_labels.append(predicted_classes)

# print("[INFO] Calculating confusion matrix")
# confusion_matrix = create_confusion_matrix(ground_truth_labels, predicted_labels, num_classes=6)

confusion_matrix = [[]]

print("[INFO] Saving results")

test_set_result = {"confusion_matrix": confusion_matrix,
                   "ground_truth_labels": ground_truth_labels,
                   "predicted_labels": predicted_labels}

_, num_neurons, trial_set = input_directory.rsplit("_", 2)
file_name = f"classification_results_n_{num_neurons}_s_{trial_set}_t_{trial_nr}.p"
output_file = os.path.join(output_directory, file_name)

with open(output_file, "wb") as pickle_file:
    pickle.dump(test_set_result, pickle_file)

print(f"[INFO] Saved results to {output_file}")

print("[INFO] DONE")
