import numpy as np
import tensorflow as tf
import pathlib
import pickle

from rsnn_utils.rsnn import python_forward_pass


def collect_data(target_directory, down_sample_rate, num_output_channels):
    all_samples = []
    all_targets = []

    for sample_file_path in pathlib.Path(target_directory).glob("*.p"):
        if sample_file_path.name.startswith("train") or sample_file_path.name.startswith("test"):
            continue

        sample_class, sample_number = sample_file_path.stem.split('_')
        sample_class = int(sample_class)
        sample_number = int(sample_number)

        with open(sample_file_path, "rb") as sample_file:
            sample_data = pickle.load(sample_file)

        if down_sample_rate is not None:
            sample_data = sample_down(sample_data, down_sample_rate)

        sample_class_encoding = np.zeros(num_output_channels)
        sample_class_encoding[sample_class] = 1

        all_samples.append(sample_data)
        all_targets.append(sample_class_encoding)

    return np.array(all_samples), np.array(all_targets)


def normalize_data(original_data, normalization_percentile):
    top_limit = np.percentile(original_data, 100 - normalization_percentile, axis=(0, 2),
                              keepdims=True, overwrite_input=True)
    bottom_limit = np.percentile(original_data, normalization_percentile, axis=(0, 2),
                                 keepdims=True, overwrite_input=True)

    normalized_data = 2 * (original_data - bottom_limit) / (top_limit - bottom_limit) - 1
    return normalized_data, top_limit, bottom_limit


def randomize_data(data_samples, data_labels):
    """randomize the order of the samples"""

    num_samples, *_ = data_samples.shape

    randomized_indices = np.arange(num_samples)
    np.random.shuffle(randomized_indices)
    randomized_samples = data_samples[randomized_indices]
    randomized_targets = data_labels[randomized_indices]

    return randomized_samples, randomized_targets


def sample_down(original_sample, down_sample_rate):
    num_input_channels, num_time_steps = original_sample.shape
    down_sampled_num_time_steps = int(np.ceil(num_time_steps / down_sample_rate))
    down_sampled_data = np.zeros((num_input_channels, down_sampled_num_time_steps))

    for new_t, old_t in enumerate(range(0, num_time_steps, down_sample_rate)):
        down_sampled_data[:, new_t] = np.mean(original_sample[:, old_t:old_t + down_sample_rate], axis=1)

    return down_sampled_data


def train_val_test_split(samples, labels, data_split):
    train_ratio, validation_ratio, test_ratio = data_split

    num_samples, *_ = samples.shape
    num_train_samples = int(np.ceil(train_ratio * num_samples))
    num_validation_samples = int(np.ceil(validation_ratio * num_samples))

    train_data = samples[:num_train_samples]
    train_labels = labels[:num_train_samples]

    validation_data = samples[num_train_samples:num_train_samples+num_validation_samples]
    validation_labels = labels[num_train_samples:num_train_samples+num_validation_samples]

    test_data = samples[num_train_samples+num_validation_samples:]
    test_labels = labels[num_train_samples+num_validation_samples:]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def turn_into_batches(samples, labels, batch_size):
    batched_data = []
    batched_targets = []

    num_samples, *_ = samples.shape

    for i in range(0, num_samples, batch_size):
        batch_data = samples[i: i + batch_size]
        batch_data = batch_data.transpose(2, 0, 1)

        batch_targets = labels[i: i + batch_size]

        batched_data.append(batch_data)
        batched_targets.append(batch_targets)

    return batched_data, batched_targets


def find_class_with_max_probability(data_of_run):
    num_time_steps, num_output_channels = data_of_run.shape

    argmax_index = np.argmax(data_of_run)
    class_index = argmax_index % num_output_channels
    time_index = argmax_index // num_output_channels

    return class_index, time_index


def find_indices_of_max_probabilities(data_of_run):
    num_time_steps, batch_size, num_classes = data_of_run.get_shape()

    flattened_data = tf.reshape(tf.transpose(data_of_run, perm=[1, 0, 2]), [batch_size, -1])
    argmax_indices = tf.math.argmax(flattened_data, axis=1)

    time_step_indices = argmax_indices // num_classes
    neuron_indices = argmax_indices % num_classes

    resulting_indices = tf.transpose(tf.stack((time_step_indices, range(batch_size), neuron_indices)))

    return resulting_indices


def evaluate_model(W_in, W_rec, W_out, tau_membrane,
                   output_time_window, threshold_voltage, dt, num_time_steps,
                   test_samples, test_labels,
                   rsnn_forward_pass_function):

    test_batch_loss = []

    all_ground_truth_labels = []
    all_predicted_labels = []

    num_neurons = W_rec.shape[0]
    num_samples = 0

    for batch_data, batch_labels in zip(test_samples, test_labels):
        current_batch_size = batch_data.shape[1]

        resulting_voltages, resulting_activations = rsnn_forward_pass_function(W_in, W_rec, tau_membrane,
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

        all_ground_truth_labels.extend(ground_truth_classes.numpy())
        all_predicted_labels.extend(predicted_classes.numpy())

        num_samples += current_batch_size
        test_batch_loss.append(batch_cost * current_batch_size)

    test_set_loss = np.sum(test_batch_loss) / num_samples

    all_ground_truth_labels = np.array(all_ground_truth_labels)
    all_predicted_labels = np.array(all_predicted_labels)
    test_set_accuracy = np.mean(all_ground_truth_labels == all_predicted_labels)

    return test_set_accuracy, test_set_loss


def evaluate_model_old(W_in, W_rec, W_out, tau_membrane,
                   output_time_window, threshold_voltage, dt, num_time_steps,
                   test_samples, test_labels,
                   rsnn_forward_pass_function):

    test_batch_accuracies = []
    test_batch_loss = []

    num_neurons = W_rec.shape[0]

    for batch_data, batch_labels in zip(test_samples, test_labels):
        current_batch_size = batch_data.shape[1]

        resulting_voltages, resulting_activations = rsnn_forward_pass_function(W_in, W_rec, tau_membrane,
                                                                               batch_data,
                                                                               threshold_voltage=threshold_voltage,
                                                                               delta_t=dt)

        smoothed_spikes = tf.stack([tf.math.reduce_mean(resulting_activations[i - output_time_window: i], axis=0)
                                    if i >= output_time_window else tf.zeros(shape=[current_batch_size, num_neurons])
                                    for i in range(1, num_time_steps+1)])

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

        test_batch_accuracies.append(batch_accuracy)
        test_batch_loss.append(batch_cost)

    test_set_accuracy = np.mean(test_batch_accuracies)
    test_set_loss = np.mean(test_batch_loss)

    return test_set_accuracy, test_set_loss
