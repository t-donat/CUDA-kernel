import os
import pathlib
import argparse
import pickle
import mne
import json
import numpy as np

parser = argparse.ArgumentParser(description='Create a Datasset out of .gdf files')
parser.add_argument("--dest",
                    dest="target_directory",
                    help="Directory to search for .gdf files")

event_annotations = {0x308: "supination_class_cue",
                     0x309: "pronation_class_cue",
                     0x30B: "hand_open_class_cue",
                     0x39D: "palmar_grasp_class_cue",
                     0x39E: "lateral_grasp_class_cue",
                     0x39F: "rest_class_clue"}

class_cue_conversion_table = {0x308: 0,
                              0x309: 1,
                              0x30B: 2,
                              0x39D: 3,
                              0x39E: 4,
                              0x39F: 5}

class_cue_count_table = {0x308: 0,
                         0x309: 0,
                         0x30B: 0,
                         0x39D: 0,
                         0x39E: 0,
                         0x39F: 0}


def verify_entries(trial_start_code, class_cue_code, trial_end_code):
    trial_start_ok = (trial_start_code == 0x300)
    class_cue_ok = (class_cue_code in [0x308, 0x309, 0x30B, 0x39D, 0x39E])
    trial_end_ok = (trial_end_code == 0x8300)

    return trial_start_ok and class_cue_ok and trial_end_ok


def extract_data(participant_run_path, data_set_directory_path):
    global class_cue_conversion_table
    global class_cue_count_table

    data_set = mne.io.read_raw_gdf(participant_run_path)
    run_start_time = data_set.annotations.orig_time.replace(tzinfo=None)

    data_info = data_set.info
    raw_data = data_set.get_data()

    lowpass_frequency = data_info["lowpass"]
    sampling_frequency = data_info["sfreq"]

    cue_annotations = data_set.annotations.to_data_frame()
    num_entries, _ = cue_annotations.shape

    time_steps = []
    input_channels = []

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

        trial_end_time = trial_end_entry.loc['onset']
        trial_end_time_delta = trial_end_time - run_start_time
        trial_end_index = int(trial_end_time_delta.total_seconds() * sampling_frequency)

        entries_ok = verify_entries(trial_start_code, class_cue_code, trial_end_code)

        if not entries_ok:
            # print("Not ok")
            continue

        sample_data = raw_data[:, trial_start_index:trial_end_index]
        sample_class = class_cue_conversion_table[class_cue_code]
        sample_number = class_cue_count_table[class_cue_code]

        class_cue_count_table[class_cue_code] += 1

        file_name = f"{sample_class}_{sample_number}.p"

        with open(os.path.join(data_set_directory_path, file_name), mode="wb") as pickle_file:
            pickle.dump(sample_data, pickle_file)

        num_input_channels, num_time_steps = sample_data.shape

        time_steps.append(num_time_steps)
        input_channels.append(num_input_channels)
        # array_lengths.append(data_length)

    return time_steps, input_channels


def extract_rest_data(participant_run_path, data_set_directory_path):
    global class_cue_conversion_table
    global class_cue_count_table

    time_steps = []
    input_channels = []

    data_set = mne.io.read_raw_gdf(participant_run_path)

    data_info = data_set.info
    raw_data = data_set.get_data()

    num_eeg_channels, num_time_steps = raw_data.shape

    sampling_frequency = data_info["sfreq"]

    expected_sample_length = int(5 * sampling_frequency)
    time_step_buffer = 0  # buffer between samples
    step_size = expected_sample_length + time_step_buffer
    assert expected_sample_length == 1280, "Sample length is incorrect"

    class_cue_code = 0x39F
    sample_class = class_cue_conversion_table[class_cue_code]  # == 5

    for i in range(0, num_time_steps, step_size):
        current_sample = raw_data[:, i: i+expected_sample_length]
        actual_input_channels, actual_sample_length = current_sample.shape

        if actual_sample_length != expected_sample_length:
            continue

        time_steps.append(actual_sample_length)
        input_channels.append(actual_input_channels)

        sample_number = class_cue_count_table[class_cue_code]

        if sample_number in [100, 1000, 2000, 3000]:
            print("wow")

        class_cue_count_table[class_cue_code] += 1

        file_name = f"{sample_class}_{sample_number}.p"

        with open(os.path.join(data_set_directory_path, file_name), mode="wb") as pickle_file:
            pickle.dump(current_sample, pickle_file)

    return time_steps, input_channels


if __name__ == '__main__':
    args = parser.parse_args()
    target_directory = args.target_directory

    data_set_directory = os.path.join(target_directory, "dataset")
    os.makedirs(data_set_directory, exist_ok=True)

    # all_array_lengths = []
    all_time_steps = []
    all_input_channels = []
    hyperparameters = {}
    first_run = True

    for participant_directory in pathlib.Path(target_directory).glob('P*'):

        for participant_run in participant_directory.glob("*.gdf"):

            if np.any([run_name in participant_run.name for run_name in ["Run 1.gdf",
                                                                         "Run 2.gdf",
                                                                         "Run 8.gdf",
                                                                         "Run 9.gdf",
                                                                         "Run 14.gdf",
                                                                         "Run 15.gdf"]]):
                time_steps, input_channels = extract_rest_data(participant_run, data_set_directory)

            else:
                time_steps, input_channels = extract_data(participant_run, data_set_directory)

            all_time_steps.extend(time_steps)
            all_input_channels.extend(input_channels)

            if first_run:
                data_set = mne.io.read_raw_gdf(participant_run)
                hyperparameters['lowpass_frequency'] = data_set.info["lowpass"]
                hyperparameters['sampling_frequency'] = data_set.info["sfreq"]

                first_run = False
        # all_array_lengths.extend(current_array_lengths)

    unique_num_time_steps = np.unique(all_time_steps)
    unique_input_channels = np.unique(all_input_channels)

    assert len(unique_num_time_steps) == 1
    assert len(unique_input_channels) == 1

    hyperparameters['num_classes'] = int(np.max(list(class_cue_conversion_table.values()))) + 1
    hyperparameters['num_time_steps'] = int(unique_num_time_steps[0])
    hyperparameters['num_input_channels'] = int(unique_input_channels[0])

    with open(os.path.join(target_directory,
                           "dataset",
                           "hyperparameters.json"), "w") as hyperparameter_file:
        json.dump(hyperparameters, hyperparameter_file, indent=4)

    # with open("log.p", "wb") as pickle_file:
    #    pickle.dump(all_array_lengths, pickle_file)

    print("Class distribution:")

    for class_cue, class_count in class_cue_count_table.items():
        class_number = class_cue_conversion_table[class_cue]
        class_name = event_annotations[class_cue]
        print(f"Class {class_number} ({class_name}): {class_count}")

    print(f"\nExtracted {np.sum(list(class_cue_count_table.values()))} samples in total")
    print(f"Dataset created and saved at {os.path.abspath(os.path.join(target_directory, 'dataset'))}")

