import numpy as np
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
    top_limit = np.percentile(original_data, 100 - normalization_percentile, axis=(0, 2), keepdims=True)
    bottom_limit = np.percentile(original_data, normalization_percentile, axis=(0, 2), keepdims=True)

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


def train_test_split(samples, labels, data_split):
    if data_split.isdigit():
        num_test_samples = int(data_split)

    else:
        test_ratio = float(data_split)
        num_samples, *_ = samples.shape
        num_test_samples = int(np.ceil(test_ratio * num_samples))

    train_data = samples[:-num_test_samples]
    train_labels = labels[:-num_test_samples]

    test_data = samples[-num_test_samples:]
    test_labels = labels[-num_test_samples:]

    return train_data, train_labels, test_data, test_labels


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


def evaluate_model(W_in, W_rec, W_out, tau_membrane,
                   output_time_window, threshold_voltage, dt, num_time_steps,
                   test_samples, test_labels):
    predictions_on_test_set = []

    for batch_data, batch_labels in zip(test_samples, test_labels):
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

        for b in range(current_batch_size):
            predicted_class, _ = find_class_with_max_probability(softmax_output[:, b])
            ground_truth_distribution = batch_labels[b]

            prediction_correct = ground_truth_distribution[predicted_class] == 1

            predictions_on_test_set.append(prediction_correct)

        test_set_accuracy = np.mean(predictions_on_test_set)

        return test_set_accuracy