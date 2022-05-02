import os
import pathlib
import argparse
import pickle
import mne
import json
import numpy as np

from rsnn_utils.preprocessing import extract_data, extract_rest_data
from rsnn_utils.preprocessing import run_preprocessing
from rsnn_utils.data import randomize_data, train_val_test_split, turn_into_batches

parser = argparse.ArgumentParser(description='Create a Datasset out of .gdf files')
parser.add_argument("--dest",
                    dest="target_directory",
                    help="Directory to search for .gdf files")

parser.add_argument('-b', '--batch_size',
                    type=int, default=32,
                    help="How many samples are contained in a batch")

parser.add_argument("-r", "--down_sample_rate",
                    type=int,
                    dest="down_sample_rate",
                    help="Rate by which to sample down the data during preprocessing")

parser.add_argument("-v", "--explained_variance",
                    type=float, default=0.99,
                    dest="explained_variance",
                    help="How much variance should be explained by the principal components that are kept")

parser.add_argument("-s", "--data_split",
                    nargs=3,
                    default=[0.8, 0.1, 0.1],
                    dest="data_split",
                    type=float,
                    help="How to split the data into train, validation and test sets")


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

runs_to_skip = []  # "P06 Run 1"

# class_cue_count_table = {0x308: 0,
#                         0x309: 0,
#                         0x30B: 0,
#                         0x39D: 0,
#                         0x39E: 0,
#                         0x39F: 0}


if __name__ == '__main__':
    args = parser.parse_args()
    target_directory = args.target_directory

    batch_size = args.batch_size
    down_sample_rate = args.down_sample_rate
    explained_variance = args.explained_variance
    data_split = args.data_split

    data_set_directory = os.path.join(target_directory, "dataset")
    os.makedirs(data_set_directory, exist_ok=True)

    all_samples = []
    all_targets = []

    all_time_steps = []
    all_input_channels = []

    hyperparameters = {}
    first_run = True

    for participant_directory in pathlib.Path(target_directory).glob('P*'):

        for participant_run_path in participant_directory.glob("*.gdf"):

            if participant_run_path.stem in runs_to_skip:
                continue

            base_name_no_ext = participant_run_path.stem
            current_participant, _, current_run = base_name_no_ext.split(" ")
            current_participant = int(current_participant[1:])
            current_run = int(current_run)

            data_set = mne.io.read_raw_gdf(participant_run_path, verbose=False)

            data_info = data_set.info
            raw_data = data_set.get_data()

            num_eeg_channels, num_time_steps = raw_data.shape

            sampling_frequency = data_info["sfreq"]
            annotations_of_run = data_set.annotations

            preprocessed_data = run_preprocessing(raw_data,
                                                  explained_variance, sampling_frequency,
                                                  down_sample_rate=down_sample_rate)

            if current_run in [1, 2, 8, 9, 14, 15]:

                (current_samples, current_targets,
                 time_steps, input_channels) = extract_rest_data(preprocessed_data,
                                                                 sampling_frequency,
                                                                 conversion_table=class_cue_conversion_table)

            else:
                (current_samples, current_targets,
                 time_steps, input_channels) = extract_data(preprocessed_data,
                                                            sampling_frequency, annotations_of_run,
                                                            conversion_table=class_cue_conversion_table)

            all_samples.extend(current_samples)
            all_targets.extend(current_targets)
            all_time_steps.extend(time_steps)
            all_input_channels.extend(input_channels)

            if first_run:
                hyperparameters['lowpass_frequency'] = data_info["lowpass"]
                hyperparameters['sampling_frequency'] = data_info["sfreq"]

                first_run = False

    all_samples = np.asarray(all_samples)
    all_targets = np.asarray(all_targets)

    class_counts = np.sum(all_targets, axis=0).astype(int)

    unique_num_time_steps = np.unique(all_time_steps)
    unique_input_channels = np.unique(all_input_channels)

    assert len(unique_num_time_steps) == 1
    assert len(unique_input_channels) == 1

    hyperparameters['num_classes'] = int(np.max(list(class_cue_conversion_table.values()))) + 1
    hyperparameters['num_time_steps'] = int(unique_num_time_steps[0])
    hyperparameters['num_input_channels'] = int(unique_input_channels[0])

    hyperparameters['explained_variance'] = explained_variance
    hyperparameters['down_sample_rate'] = down_sample_rate
    hyperparameters["batch_size"] = batch_size

    print("Class distribution:")

    for class_cue, class_count in zip(class_cue_conversion_table.keys(), class_counts):
        class_number = class_cue_conversion_table[class_cue]
        class_name = event_annotations[class_cue]
        print(f"Class {class_number} ({class_name}): {class_count}")

    print(f"\n[INFO] Extracted {np.sum(class_counts)} samples in total")

    print("[INFO] Randomizing data")
    randomized_samples, randomized_targets = randomize_data(all_samples, all_targets)

    print("[INFO] Splitting into train, validation and test sets")

    (train_data, train_labels,
     validation_data, validation_labels,
     test_data, test_labels) = train_val_test_split(randomized_samples, randomized_targets, data_split)

    num_train_samples, *_ = train_data.shape
    print(f"Train: {num_train_samples}")
    assert train_labels.shape[0] == num_train_samples, f"{train_labels.shape} != {train_data.shape}"

    num_validation_samples, *_ = validation_data.shape
    print(f"Validation: {num_validation_samples}")
    assert validation_labels.shape[0] == num_validation_samples, f"{validation_labels.shape} != {validation_data.shape}"

    num_test_samples, *_ = test_data.shape
    print(f"Test: {num_test_samples}")
    assert test_labels.shape[0] == num_test_samples, f"{test_labels.shape} != {test_data.shape}"

    print(f"[INFO] Organizing the sets into batches of {batch_size} samples")

    train_data_set = turn_into_batches(train_data, train_labels, batch_size)
    validation_data_set = turn_into_batches(validation_data, validation_labels,  batch_size)
    test_data_set = turn_into_batches(test_data, test_labels, batch_size)

    print("[INFO] Saving the data sets")

    with open(os.path.join(data_set_directory, "hyperparameters.json"), "w") as file:
        json.dump(hyperparameters, file, indent=4)

    with open(os.path.join(data_set_directory, "train_data_set.p"), "wb") as pickle_file:
        pickle.dump(train_data_set, pickle_file)

    with open(os.path.join(data_set_directory, "validation_data_set.p"), "wb") as pickle_file:
        pickle.dump(validation_data_set, pickle_file)

    with open(os.path.join(data_set_directory, "test_data_set.p"), "wb") as pickle_file:
        pickle.dump(test_data_set, pickle_file)

    print(f"Dataset created and saved at {os.path.abspath(data_set_directory)}")

