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

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

initial_membrane_time_constant = 200 / 1000

output_time_window = 100
threshold_voltage = 1
expected_firing_rate = 0.2

gradient_scaling_factor = 0.3

learning_rate = 0.001
weight_decay_rate = 0.0001
firing_rate_lambda = 1000.0
time_constant_lambda = 1.0

network_dropout_rate = 0.3
input_dropout_rate = 0.3
gradient_clipping_value = 1_000

train_time_constants = False

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

verbose = True
debug = False

# -------------------------------------------------------------------
# GPU SETUP
# -------------------------------------------------------------------

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

# parser.add_argument("--val",
#                    help="Whether to expect and use a validation set",
#                    action="store_true",
#                    default=False,
#                    dest="use_validation_set")

arguments = parser.parse_args()
input_directory = arguments.input_directory
output_directory = arguments.output_directory
job_id = arguments.job_id
num_epochs = int(arguments.num_epochs)
num_neurons = int(arguments.num_neurons)

with open(os.path.join(input_directory, "train_data_set.p"), "rb") as pickle_file:
    train_data_set = pickle.load(pickle_file)

val_set_file_path = os.path.join(input_directory, "validation_data_set.p")
if os.path.exists(val_set_file_path):
    print("Found validation set")
    use_validation_set = True

    with open(val_set_file_path, "rb") as pickle_file:
        validation_data_set = pickle.load(pickle_file)

    validation_samples, validation_labels = validation_data_set
    del validation_data_set

else:
    use_validation_set = False

with open(os.path.join(input_directory, "hyperparameters.json"), "r") as file:
    hyperparameters = json.load(file)

sampling_frequency = hyperparameters["sampling_frequency"]
num_classes = hyperparameters["num_classes"]
num_time_steps = hyperparameters["num_time_steps"]
num_input_channels = hyperparameters["num_input_channels"]

train_samples, train_labels = train_data_set
del train_data_set

# -------------------------------------------------------------------
# SAVING THE HYPERPARAMETERS
# -------------------------------------------------------------------

# num_neurons = 128

dt = 1 / sampling_frequency

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
hyperparameters["time_constant_lambda"] = time_constant_lambda

hyperparameters["network_dropout_rate"] = network_dropout_rate
hyperparameters["input_dropout_rate"] = input_dropout_rate

hyperparameters["gradient_clipping_value"] = gradient_clipping_value
hyperparameters["train_time_constants"] = train_time_constants


# -------------------------------------------------------------------
# LOAD IN CUDA FUNCTIONS
# -------------------------------------------------------------------
shared_lib_path = os.path.join(os.path.dirname(__file__), "spiking_network.so")

if not os.path.exists(shared_lib_path):
    raise FileNotFoundError(f"Could not find shared library at expected path: {shared_lib_path}")
rsnn = tf.load_op_library(shared_lib_path)

# -------------------------------------------------------------------
# SET UP VARIABLES
# -------------------------------------------------------------------

(W_in_init, W_rec_init,
 W_out_init, tau_membrane_init) = initialize_weights(num_neurons, num_input_channels, num_classes,
                                                     threshold_voltage, initial_membrane_time_constant)

with tf.device("/cpu"):
    W_in = tf.Variable(W_in_init, dtype=tf.float32)
    W_rec = tf.Variable(W_rec_init, dtype=tf.float32)
    W_out = tf.Variable(W_out_init, dtype=tf.float32)
    tau_membrane = tf.Variable(tau_membrane_init, dtype=tf.float32)

optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay_rate)

if train_time_constants:
    variables = [W_in, W_rec, W_out, tau_membrane]
else:
    variables = [W_in, W_rec, W_out]

train_samples_as_tensors, train_batch_sizes = convert_batch_to_tensors(train_samples)
del train_samples

debug_interval = int(np.ceil(num_epochs / 10))

training_stats = {"accuracy": [],
                  "overall_loss": [],
                  "cross_entropy_loss": [],
                  "fire_rate_loss": [],
                  "time_constant_loss": []}

best_performing_parameters = {"val_accuracy": 0.0,
                              "W_in": W_in,
                              "W_rec": W_rec,
                              "W_out": W_out,
                              "tau_membrane": tau_membrane}

validation_stats = {"accuracy": [],
                    "loss": [],
                    "best_performing_parameters": None}

firing_rates_over_epochs = []
time_constants_over_epochs = []
all_global_norms = []


if debug:
    time_constant_derivatives_over_epochs = []

    w_in_stats = {"mean": [],
                  "std": [],
                  "norm": []}

    w_rec_stats = {"mean": [],
                   "std": [],
                   "norm": []}

    w_out_stats = {"mean": [],
                   "std": [],
                   "norm": []}

    w_in_derivative_stats = {"mean": [],
                             "std": [],
                             "norm": [],
                             "clipped_norm": []}

    w_rec_derivative_stats = {"mean": [],
                              "std": [],
                              "norm": [],
                              "clipped_norm": []}

    w_out_derivative_stats = {"mean": [],
                              "std": [],
                              "norm": [],
                              "clipped_norm": []}

    tau_derivative_stats = {"mean": [],
                            "std": [],
                            "norm": [],
                            "clipped_norm": []}

    overshooting_batches = {"epoch": [],
                            "batch": [],
                            "w_in": [],
                            "w_rec": [],
                            "w_out": [],
                            "tau_membrane": [],
                            "w_in_derivative": [],
                            "w_rec_derivative": [],
                            "w_out_derivative": [],
                            "tau_membrane_derivative": []}

# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------

print(f"Starting training for {num_epochs} epochs")

start = time.time()

for current_epoch in range(1, num_epochs + 1):

    loss_in_epoch = []
    cross_entropy_loss_in_epoch = []
    fire_rate_loss_in_epoch = []
    time_constant_loss_in_epoch = []
    accuracy_in_epoch = []

    for batch_number, (batch_data, batch_labels) in enumerate(zip(train_samples_as_tensors, train_labels)):
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
                                    for i in range(1, num_time_steps+1)])

        network_output = tf.linalg.matmul(smoothed_spikes, W_out, transpose_b=True)

        softmax_output = tf.math.exp(network_output - tf.math.reduce_max(network_output))
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

        time_constant_loss = time_constant_lambda * tf.math.reduce_mean(
            (tau_membrane - initial_membrane_time_constant)**2).numpy()

        if tf.math.is_nan(time_constant_loss):
            print(f"NaN in epoch {current_epoch}, batch {batch_number}")

        batch_cost = original_batch_cost + firing_rate_loss + time_constant_loss

        loss_in_epoch.append(batch_cost)

        cross_entropy_loss_in_epoch.append(original_batch_cost)
        fire_rate_loss_in_epoch.append(firing_rate_loss)
        time_constant_loss_in_epoch.append(time_constant_loss)

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
                                   for i in range(1, num_time_steps+1)])

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
         original_dE_dtau_membrane) = rsnn.backward_pass(partial_dE_dv, W_rec, tau_membrane,
                                                         dropout_corrected_batch_data,
                                                         resulting_voltages, dropout_corrected_activations,
                                                         threshold_voltage=threshold_voltage, delta_t=dt,
                                                         gradient_scaling_factor=gradient_scaling_factor)

        membrane_time_constant_dE_dz = time_constant_lambda / num_neurons * (tau_membrane - initial_membrane_time_constant)

        dE_dtau_membrane = original_dE_dtau_membrane + membrane_time_constant_dE_dz

        if train_time_constants:
            gradients = [dE_dW_in, dE_dW_rec, dE_dW_out, dE_dtau_membrane]
        else:
            gradients = [dE_dW_in, dE_dW_rec, dE_dW_out]

        clipped_gradients = [tf.clip_by_norm(g, gradient_clipping_value) for g in gradients]
        global_norm = tf.linalg.global_norm(gradients)

        # clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, gradient_clipping_value)
        global_norm = global_norm.numpy()

        optimizer.apply_gradients(zip(clipped_gradients, variables))

        time_constants_over_epochs.append(tau_membrane.numpy().flatten())
        all_global_norms.append(global_norm)

        if debug:
            time_constant_derivatives_over_epochs.append(dE_dtau_membrane.numpy().flatten())
            # Weight Statistics
            # W_in
            w_in_stats["mean"].append(tf.math.reduce_mean(W_in).numpy())
            w_in_stats["std"].append(tf.math.reduce_std(W_in).numpy())
            w_in_norm = tf.norm(W_in).numpy()
            w_in_stats["norm"].append(w_in_norm)

            # W_rec
            w_rec_stats["mean"].append(tf.math.reduce_mean(W_rec).numpy())
            w_rec_stats["std"].append(tf.math.reduce_std(W_rec).numpy())
            w_rec_norm = tf.norm(W_rec).numpy()
            w_rec_stats["norm"].append(w_rec_norm)

            # W_out
            w_out_stats["mean"].append(tf.math.reduce_mean(W_out).numpy())
            w_out_stats["std"].append(tf.math.reduce_std(W_out).numpy())
            w_out_norm = tf.norm(W_out).numpy()
            w_out_stats["norm"].append(w_out_norm)

            # Weight Derivative Statistics
            if train_time_constants:
                clipped_dE_dW_in, clipped_dE_dW_rec, clipped_dE_dW_out, clipped_tau_derivative = clipped_gradients
            else:
                clipped_dE_dW_in, clipped_dE_dW_rec, clipped_dE_dW_out = clipped_gradients
                clipped_tau_derivative = tf.zeros(shape=[num_neurons, 1])

            w_in_derivative_norm = tf.norm(dE_dW_in).numpy()
            clipped_w_in_derivative_norm = tf.norm(clipped_dE_dW_in).numpy()

            w_rec_derivative_norm = tf.norm(dE_dW_rec).numpy()
            clipped_w_rec_derivative_norm = tf.norm(clipped_dE_dW_rec).numpy()

            w_out_derivative_norm = tf.norm(dE_dW_out).numpy()
            clipped_w_out_derivative_norm = tf.norm(clipped_dE_dW_out).numpy()

            tau_derivative_norm = tf.norm(dE_dtau_membrane).numpy()
            #clipped_tau_derivative_norm = tf.norm(tau_derivative_norm).numpy()
            clipped_tau_derivative_norm = tf.norm(clipped_tau_derivative).numpy()

            # W_in
            w_in_derivative_stats["mean"].append(tf.math.reduce_mean(clipped_dE_dW_in).numpy())
            w_in_derivative_stats["std"].append(tf.math.reduce_std(clipped_dE_dW_in).numpy())
            w_in_derivative_stats["norm"].append(w_in_derivative_norm)
            w_in_derivative_stats["clipped_norm"].append(clipped_w_in_derivative_norm)

            # W_rec
            w_rec_derivative_stats["mean"].append(tf.math.reduce_mean(clipped_dE_dW_rec).numpy())
            w_rec_derivative_stats["std"].append(tf.math.reduce_std(clipped_dE_dW_rec).numpy())
            w_rec_derivative_stats["norm"].append(w_rec_derivative_norm)
            w_rec_derivative_stats["clipped_norm"].append(clipped_w_rec_derivative_norm)

            # W_out
            w_out_derivative_stats["mean"].append(tf.math.reduce_mean(clipped_dE_dW_out).numpy())
            w_out_derivative_stats["std"].append(tf.math.reduce_std(clipped_dE_dW_out).numpy())
            w_out_derivative_stats["norm"].append(w_out_derivative_norm)
            w_out_derivative_stats["clipped_norm"].append(clipped_w_out_derivative_norm)

            # tau
            tau_derivative_stats["mean"].append(tf.math.reduce_mean(clipped_tau_derivative).numpy())
            tau_derivative_stats["std"].append(tf.math.reduce_std(clipped_tau_derivative).numpy())
            tau_derivative_stats["norm"].append(tau_derivative_norm)
            tau_derivative_stats["clipped_norm"].append(clipped_tau_derivative_norm)

            current_norms = np.array((w_in_derivative_norm, w_rec_derivative_norm, w_out_derivative_norm))
            if np.any(current_norms > gradient_clipping_value):
                # print("test")
                overshooting_batches["epoch"].append(current_epoch)
                overshooting_batches["batch"].append(batch_number)

                overshooting_batches["w_in"].append(W_in.numpy())
                overshooting_batches["w_rec"].append(W_rec.numpy())
                overshooting_batches["w_out"].append(W_out.numpy())
                overshooting_batches["tau_membrane"].append(tau_membrane.numpy())

                overshooting_batches["w_in_derivative"].append(dE_dW_in.numpy())
                overshooting_batches["w_rec_derivative"].append(dE_dW_rec.numpy())
                overshooting_batches["w_out_derivative"].append(dE_dW_out.numpy())
                overshooting_batches["tau_membrane_derivative"].append(dE_dtau_membrane.numpy())

    avg_overall_loss = tf.math.reduce_mean(loss_in_epoch).numpy()
    avg_cross_entropy_loss = tf.math.reduce_mean(cross_entropy_loss_in_epoch).numpy()
    avg_fire_rate_loss = tf.math.reduce_mean(fire_rate_loss_in_epoch).numpy()
    avg_time_constant_loss = tf.math.reduce_mean(time_constant_loss_in_epoch).numpy()
    avg_accuracy = tf.math.reduce_mean(accuracy_in_epoch).numpy()

    training_stats["accuracy"].append(avg_accuracy)
    training_stats["overall_loss"].append(avg_overall_loss)
    training_stats["cross_entropy_loss"].append(avg_cross_entropy_loss)
    training_stats["fire_rate_loss"].append(avg_fire_rate_loss)
    training_stats["time_constant_loss"].append(avg_time_constant_loss)

    if use_validation_set:
        validation_set_accuracy,  validation_set_loss = evaluate_model(W_in, W_rec, W_out, tau_membrane,
                                                                       output_time_window, threshold_voltage,
                                                                       dt, num_time_steps,
                                                                       validation_samples, validation_labels,
                                                                       rsnn.forward_pass)

        validation_stats["loss"].append(validation_labels)
        validation_stats["accuracy"].append(validation_set_accuracy)

        if validation_set_accuracy > best_performing_parameters["val_accuracy"]:
            best_performing_parameters["val_accuracy"] = validation_set_accuracy
            best_performing_parameters["W_in"] = W_in
            best_performing_parameters["W_rec"] = W_rec
            best_performing_parameters["W_out"] = W_out
            best_performing_parameters["tau_membrane"] = tau_membrane

    if current_epoch == 1 or current_epoch % debug_interval == 0:

        rounded_train_loss = round(float(avg_overall_loss), 3)
        rounded_train_accuracy = round(float(avg_accuracy * 100), 2)
        print(f"\nEpoch {current_epoch}:")
        print(f"Train: Loss: {rounded_train_loss}, Accuracy: {rounded_train_accuracy}%")

        if use_validation_set:
            rounded_validation_loss = round(float(validation_set_loss), 3)
            rounded_validation_accuracy = round(float(validation_set_accuracy * 100), 2)
            print(f"Validation: Loss: {rounded_validation_loss}, Accuracy: {rounded_validation_accuracy}%")

        if verbose:
            print("\nLosses:")
            print(f"Cross entropy loss: {avg_cross_entropy_loss}")
            print(f"Fire rate loss: {avg_fire_rate_loss}")
            print(f"Time constant loss: {avg_time_constant_loss}")

        if debug:
            print("\nDerivatives:")
            print(f"Global norm: {global_norm}")
            print(f"W_in norm: {w_in_derivative_norm}")
            print(f"W_rec norm: {w_rec_derivative_norm}")
            print(f"W_out norm: {w_out_derivative_norm}")
            print(f"Tau norm: {tau_derivative_norm}")

        # print(tf.math.reduce_mean(membrane_time_constant_dE_dz).numpy())

training_duration = time.time() - start

print("Training completed")

del train_samples_as_tensors
del train_labels

if use_validation_set:
    del validation_samples
    del validation_labels

# -------------------------------------------------------------------
# TESTING
# -------------------------------------------------------------------

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

print("\nRESULTS")
print("Test:")
print(f"Accuracy: {round(test_set_accuracy * 100, 3)}%, Loss: {round(test_set_loss, 3)}")
print("Validation:")
print(f"Max. accuracy: {round(best_performing_parameters['val_accuracy'] * 100, 3)}")


# -------------------------------------------------------------------
# PROCESSING RESULTS
# -------------------------------------------------------------------


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


validation_stats["best_performing_parameters"] = best_performing_parameters

last_train_accuracy = round(training_stats["accuracy"][-1] * 100, 2)
highest_train_accuracy = round(np.max(training_stats["accuracy"]) * 100, 2)
test_set_accuracy = round(test_set_accuracy * 100, 2)
duration_time_string = convert_seconds_to_dhms(training_duration)

overview = {"train_accuracy": last_train_accuracy,
            "max_train_accuracy": highest_train_accuracy,
            "test_accuracy": test_set_accuracy,
            "test_loss": test_set_loss,
            "training_duration": duration_time_string}

extra_info = {"train_batch_sizes": train_batch_sizes,
              "test_batch_sizes": test_batch_sizes}

history = {"training": training_stats,
           "overview": overview,
           "hyperparameters": hyperparameters,
           "extra_info": extra_info}

if use_validation_set:
    history["validation"] = validation_stats


# -------------------------------------------------------------------
# SAVING RESULTS
# -------------------------------------------------------------------

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

with open(os.path.join(output_directory, "history.p"), "wb") as pickle_file:
    pickle.dump(history, pickle_file)


# EXTRA FILES
with open(os.path.join(output_directory, "firing_rates_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(firing_rates_over_epochs, pickle_file)

with open(os.path.join(output_directory, "time_constants_over_epochs.p"), "wb") as pickle_file:
    pickle.dump(np.array(time_constants_over_epochs), pickle_file)

with open(os.path.join(output_directory, "all_global_norms.p"), "wb") as pickle_file:
    pickle.dump(all_global_norms, pickle_file)

if debug:
    weight_stats = {"w_in": w_in_stats,
                    "w_rec": w_rec_stats,
                    "w_out": w_out_stats}

    derivative_stats = {"w_in": w_in_derivative_stats,
                        "w_rec": w_rec_derivative_stats,
                        "w_out": w_out_derivative_stats,
                        "tau": tau_derivative_stats}

    debug_stats = {"weights": weight_stats,
                   "derivatives": derivative_stats,
                   "overshooting_batches": overshooting_batches}

    with open(os.path.join(output_directory, "debug_stats.p"), "wb") as pickle_file:
        pickle.dump(debug_stats, pickle_file)

    with open(os.path.join(output_directory, "time_constant_derivatives_over_epochs.p"), "wb") as pickle_file:
        pickle.dump(np.array(time_constant_derivatives_over_epochs), pickle_file)

