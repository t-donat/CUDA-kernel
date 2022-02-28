import os
import pathlib
import argparse
import pickle
import json
import numpy as np

from rsnn_utils.data import collect_data, normalize_data, randomize_data, train_val_test_split, turn_into_batches

parser = argparse.ArgumentParser(description='Load a preprocessed Data Set')
parser.add_argument("-i", "--input_dir",
                    dest="target_directory",
                    help="Directory to search for pickle files")

parser.add_argument('-b', '--batch_size',
                    type=int, default=32,
                    help="How many samples are contained in a batch")

parser.add_argument("-r", "--down_sample_rate",
                    type=int,
                    dest="down_sample_rate",
                    help="Rate by which to sample down the data during preprocessing")

parser.add_argument("-p", "--percentile",
                    type=int, default=2,
                    dest="percentile",
                    help="Percentile from the top and bottom to use for normalization")

parser.add_argument("-s", "--data_split",
                    nargs=3,
                    default=[0.8, 0.1, 0.1],
                    dest="data_split",
                    type=float,
                    help="How to split the data into train, validation and test sets")

class_cue_conversion_table = {0x308: 0,
                              0x309: 1,
                              0x30B: 2,
                              0x39D: 3,
                              0x39E: 4,
                              0x39F: 5}


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    target_directory = args.target_directory
    down_sample_rate = args.down_sample_rate
    normalization_percentile = args.percentile
    data_split = args.data_split

    if not os.path.isdir(target_directory):
        raise NotADirectoryError(f"Target directory {target_directory} does not exist")

    print(f"[INFO] Searching target directory {os.path.abspath(target_directory)} for pickle files.")

    data_sample_files = pathlib.Path(target_directory).glob("*.p")

    num_samples = len(list(data_sample_files))
    num_classes = np.max(list(class_cue_conversion_table.values())) + 1

    if num_samples == 0:
        raise FileNotFoundError(f"No pickle files found at target directory {target_directory}")
    else:
        print(f'[INFO] Found {num_samples} samples')

    with open(os.path.join(target_directory, "hyperparameters.json"), "r") as file:
        hyperparameters = json.load(file)

    print("[INFO] Collecting samples")
    all_samples, all_targets = collect_data(target_directory, down_sample_rate, num_classes)

    if down_sample_rate is not None:
        *_, new_num_time_steps = all_samples.shape
        hyperparameters["num_time_steps"] = int(new_num_time_steps)

    print("[INFO] Normalizing data")
    normalized_data, top_limit, bottom_limit = normalize_data(all_samples, normalization_percentile)

    del all_samples

    print("[INFO] Randomizing data")
    randomized_samples, randomized_targets = randomize_data(normalized_data, all_targets)

    del normalized_data
    del all_targets

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

    del randomized_samples
    del randomized_targets

    print(f"[INFO] Organizing the sets into batches of {batch_size} samples")

    batched_train_data, batched_train_labels = turn_into_batches(train_data, train_labels, batch_size)
    batched_validation_data, batched_validation_labels = turn_into_batches(validation_data, validation_labels, batch_size)
    batched_test_data, batched_test_labels = turn_into_batches(test_data, test_labels, batch_size)

    del train_data
    del train_labels
    del validation_data
    del validation_labels
    del test_data
    del test_labels

    train_data_set = (batched_train_data, batched_train_labels)
    validation_data_set = (batched_test_data, batched_test_labels)
    test_data_set = (batched_test_data, batched_test_labels)

    del batched_train_data
    del batched_train_labels
    del batched_validation_data
    del batched_validation_labels
    del batched_test_data
    del batched_test_labels

    hyperparameters["batch_size"] = batch_size
    hyperparameters["max_normalization_value"] = list(top_limit.flatten())
    hyperparameters["min_normalization_value"] = list(bottom_limit.flatten())

    save_directory = os.path.join(target_directory, "dataset")
    os.makedirs(save_directory, exist_ok=True)

    print("[INFO] Saving the data sets")

    with open(os.path.join(save_directory, "hyperparameters.json"), "w") as file:
        json.dump(hyperparameters, file, indent=4)

    with open(os.path.join(save_directory, "train_data_set.p"), "wb") as pickle_file:
        pickle.dump(train_data_set, pickle_file)

    with open(os.path.join(save_directory, "validation_data_set.p"), "wb") as pickle_file:
        pickle.dump(validation_data_set, pickle_file)

    with open(os.path.join(save_directory, "test_data_set.p"), "wb") as pickle_file:
        pickle.dump(test_data_set, pickle_file)

    print(f"Created dataset, saved to {os.path.abspath(save_directory)}")
