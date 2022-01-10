import os
import pathlib
import argparse
import pickle
import mne

parser = argparse.ArgumentParser(description='Create a Datasset out of .gdf files')
parser.add_argument("--dest",
                    dest="target_directory",
                    help="Directory to search for .gdf files")

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

class_cue_count_table = {0x308: 0,
                         0x309: 0,
                         0x30B: 0,
                         0x39D: 0,
                         0x39E: 0}


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

    for i in range(0, num_entries, 7):

        try:
            trial_start_entry = cue_annotations.iloc[i]
            class_cue_entry = cue_annotations.iloc[i + 3]
            trial_end_entry = cue_annotations.iloc[i + 4]

        except IndexError:
            print("Indexing problem")
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
            print("Not ok")
            continue

        sample_data = raw_data[:, trial_start_index:trial_end_index]
        sample_class = class_cue_conversion_table[class_cue_code]
        sample_number = class_cue_count_table[class_cue_code]

        class_cue_count_table[class_cue_code] += 1

        file_name = f"{sample_class}_{sample_number}.p"

        with open(os.path.join(data_set_directory_path, file_name), mode="wb") as pickle_file:
            pickle.dump(sample_data, pickle_file)


if __name__ == '__main__':
    args = parser.parse_args()
    target_directory = args.target_directory

    data_set_directory = os.path.join(target_directory, "dataset")
    os.makedirs(data_set_directory, exist_ok=True)

    for participant_directory in pathlib.Path(target_directory).glob('P*'):

        for participant_run in participant_directory.glob("*.gdf"):
            extract_data(participant_run, data_set_directory)