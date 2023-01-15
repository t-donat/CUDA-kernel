import os
import numpy as np
import pickle

from scipy.signal import butter, sosfiltfilt
# from scipy.signal import lfilter, sosfilt,

from rsnn_utils.data import sample_down


def calculate_pca_matrix(data, nr_dimensions=None, explained_variance=None):
    data = data.T

    num_time_steps, num_input_channels = data.shape

    mean_subtracted_data = data - np.mean(data, axis=0, keepdims=True)
    covariance_matrix = np.dot(mean_subtracted_data.T, mean_subtracted_data) / (num_time_steps - 1)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    descending_order_indices = np.argsort(eigenvalues)[::-1]

    if nr_dimensions is not None:
        chosen_indices = descending_order_indices[:nr_dimensions]

    if explained_variance is not None:
        total_variance = np.sum(eigenvalues)
        explained_variance_of_components = np.cumsum(eigenvalues) / total_variance

        keep_component = explained_variance_of_components <= 0.99

        if explained_variance != 1.0:
            last_component_to_keep = int(np.sum(keep_component))

            keep_component[last_component_to_keep] = True

        chosen_indices = descending_order_indices[keep_component]

    else:
        chosen_indices = descending_order_indices

    projection_matrix = eigenvectors[:, chosen_indices].T

    return projection_matrix


def low_rank_projection(data_to_project, pca_projection_matrix):
    low_rank_projection_matrix = np.dot(pca_projection_matrix.T, pca_projection_matrix)

    return np.dot(low_rank_projection_matrix.T, data_to_project)


def run_preprocessing(raw_data, explained_variance, sampling_rate, down_sample_rate=None):

    neurons_with_nan, time_steps_with_nan = np.where(np.isnan(raw_data))

    first_time_step_with_nan = np.unique(time_steps_with_nan)[0]
    no_nan_data = raw_data[:, :first_time_step_with_nan]

    sos = butter(2, Wn=[0.3, 70.0], btype="bandpass", fs=sampling_rate, output="sos")
    filtered_data = sosfiltfilt(sos, no_nan_data)

    pca_projection_matrix = calculate_pca_matrix(filtered_data, explained_variance=explained_variance)
    # ICA perhaps
    projected_data = low_rank_projection(no_nan_data, pca_projection_matrix)

    sos = butter(2, Wn=[0.3, 3.0], btype="bandpass", fs=sampling_rate, output="sos")
    # sos = butter(2, Wn=[0.3, 70.0], btype="bandpass", fs=sampling_rate, output="sos")
    processed_data = sosfiltfilt(sos, projected_data)

    if down_sample_rate is None:
        resampled_data = processed_data

    else:
        resampled_data = sample_down(processed_data, down_sample_rate)

    # Common Average Reference
    rereferenced_data = resampled_data - resampled_data.mean(axis=0, keepdims=True)

    return rereferenced_data


def verify_entries(trial_start_code, class_cue_code, trial_end_code):
    trial_start_ok = (trial_start_code == 0x300)
    class_cue_ok = (class_cue_code in [0x308, 0x309, 0x30B, 0x39D, 0x39E])
    trial_end_ok = (trial_end_code == 0x8300)

    return trial_start_ok and class_cue_ok and trial_end_ok


def extract_data(preprocessed_data,
                 sampling_frequency, annotations_of_run,
                 conversion_table, threshold_value=100):

    num_seconds_to_skip = 2.0  # at the beginning of each sample
    total_num_classes = len(conversion_table)

    run_start_time = annotations_of_run.orig_time.replace(tzinfo=None)

    cue_annotations = annotations_of_run.to_data_frame()
    num_entries, _ = cue_annotations.shape

    if num_entries % 7 != 0:
        print("Attempted movement run with irregular number of annotations!")

    time_steps_to_skip = int(num_seconds_to_skip * sampling_frequency)

    time_steps = []
    input_channels = []
    current_samples = []
    current_targets = []

    for i in range(0, num_entries, 7):

        try:
            trial_start_entry = cue_annotations.iloc[i]
            class_cue_entry = cue_annotations.iloc[i + 3]
            trial_end_entry = cue_annotations.iloc[i + 4]

        except IndexError:
            # print("Indexing problem")
            continue

        trial_start_code = int(trial_start_entry.loc['description'])
        class_cue_code = int(class_cue_entry.loc['description'])
        trial_end_code = int(trial_end_entry.loc['description'])

        trial_start_time = trial_start_entry.loc['onset']
        trial_start_time_delta = trial_start_time - run_start_time
        trial_start_index = int(trial_start_time_delta.total_seconds() * sampling_frequency)

        trial_start_index = trial_start_index + time_steps_to_skip

        trial_end_time = trial_end_entry.loc['onset']
        trial_end_time_delta = trial_end_time - run_start_time
        trial_end_index = int(trial_end_time_delta.total_seconds() * sampling_frequency)

        entries_ok = verify_entries(trial_start_code, class_cue_code, trial_end_code)

        if not entries_ok:
            # print("Not ok")
            continue

        sample_data = preprocessed_data[:, trial_start_index:trial_end_index]
        sample_class = conversion_table[class_cue_code]

        if np.any(sample_data > threshold_value):
            continue

        sample_class_encoding = np.zeros(total_num_classes)
        sample_class_encoding[sample_class] = 1

        num_input_channels, num_time_steps = sample_data.shape

        current_samples.append(sample_data)
        current_targets.append(sample_class_encoding)
        time_steps.append(num_time_steps)
        input_channels.append(num_input_channels)

    return current_samples, current_targets, time_steps, input_channels


def extract_rest_data(preprocessed_data,
                      sampling_frequency,
                      conversion_table, threshold_value=100):

    num_samples_to_skip_at_the_beginning = 500
    num_samples_to_skip_at_the_end = 500

    num_seconds_to_skip = 2.0  # at the beginning of each sample
    expected_sample_length = 3.0  # seconds
    time_step_buffer = 0  # after a sample

    total_num_classes = len(conversion_table)

    class_cue_code = 0x39F
    sample_class = conversion_table[class_cue_code]  # == 5

    cropped_data = preprocessed_data[:, num_samples_to_skip_at_the_beginning:-num_samples_to_skip_at_the_end]

    time_steps = []
    input_channels = []
    current_samples = []
    current_targets = []

    num_eeg_channels, num_time_steps = cropped_data.shape

    time_steps_to_skip = int(num_seconds_to_skip * sampling_frequency)
    time_steps_in_a_sample = int(expected_sample_length * sampling_frequency)

    step_size = time_steps_to_skip + time_steps_in_a_sample + time_step_buffer

    for i in range(0, num_time_steps, step_size):
        current_sample = cropped_data[:, i+time_steps_to_skip: i+time_steps_to_skip+time_steps_in_a_sample]
        actual_input_channels, actual_sample_length = current_sample.shape

        if actual_sample_length != time_steps_in_a_sample:
            continue

        if np.any(current_sample > threshold_value):
            continue

        sample_class_encoding = np.zeros(total_num_classes)
        sample_class_encoding[sample_class] = 1

        time_steps.append(actual_sample_length)
        input_channels.append(actual_input_channels)

        current_samples.append(current_sample)
        current_targets.append(sample_class_encoding)

    return current_samples, current_targets, time_steps, input_channels
