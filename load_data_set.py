import os
import pathlib
import argparse
import pickle
import mne
import json
import numpy as np

parser = argparse.ArgumentParser(description='Load a preprocessed Data Set')
parser.add_argument('batch_size', type=int, help="How many samples are contained in a batch")
parser.add_argument("--dest",
                    dest="target_directory",
                    help="Directory to search for pickle files")

event_annotations = {0x308: "supination_class_cue",
                     0x309: "pronation_class_cue",
                     0x30B: "hand_open_class_cue",
                     0x39D: "palmar_grasp_class_cue",
                     0x39E: "lateral_grasp_class_cue"}

class_cue_conversion_table = {0x308: 0,
                              0x309: 1,
                              0x30B: 2,
                              0x39D: 3,
                              0x39E: 4}

if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    target_directory = args.target_directory

    if not os.path.isdir(target_directory):
        raise NotADirectoryError(f"Target directory {target_directory} does not exist")

    data_sample_files = pathlib.Path(target_directory).glob("*.p")

    num_samples = len(list(data_sample_files))
    num_classes = np.max(list(class_cue_conversion_table.values()))

    if num_samples == 0:
        raise FileNotFoundError(f"No pickle files found at target directory {target_directory}")
    else:
        print(f'Found {num_samples} samples in target directory {os.path.abspath(target_directory)}')

    all_samples = []
    all_targets = []

    for sample_file_path in pathlib.Path(target_directory).glob("*.p"):
        sample_class, sample_number = sample_file_path.name.split('_')

        with open(sample_file_path, "rb") as sample_file:
            sample_data = pickle.load(sample_file)

        all_samples.append(sample_data)
        all_targets.append(sample_class)

    all_samples = np.array(all_samples)
    all_targets = np.array(all_targets)

    for i in range(0, num_samples, batch_size):
        batch_data = all_samples[i: i+batch_size]

        # print(batch_data.shape)
        batch_data = batch_data.transpose(2, 0, 1)

        batch_targets = all_targets[i: i+batch_size]
        # model classes somehow

        # save somehow

    # print(batch_data.shape)
    # print(batch_targets.shape)



