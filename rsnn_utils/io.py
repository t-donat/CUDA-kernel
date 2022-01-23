import os
import pickle


def read_pickle_files(file_directory="./data"):
    with open(os.path.join(file_directory, "hyperparameters.p"), "rb") as pickle_file:
        hyperparameters = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "input_weights.p"), "rb") as pickle_file:
        input_weights = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "recurrent_weights.p"), "rb") as pickle_file:
        recurrent_weights = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "output_weights.p"), "rb") as pickle_file:
        output_weights = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "time_series_data.p"), "rb") as pickle_file:
        time_series_data = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "resulting_voltages.p"), "rb") as pickle_file:
        resulting_voltages = pickle.load(pickle_file)

    with open(os.path.join(file_directory, "resulting_activations.p"), "rb") as pickle_file:
        resulting_activations = pickle.load(pickle_file)

    return (hyperparameters,
            input_weights, recurrent_weights, output_weights,
            time_series_data,
            resulting_voltages, resulting_activations)


def save_to_pickle_files(network_hyperparameters,
                         input_weights, recurrent_weights, membrane_time_constants, output_weights,
                         time_series_data,
                         resulting_voltages, resulting_activations,
                         expected_input_gradient, expected_recurrent_gradient,
                         resulting_input_gradient, resulting_recurrent_gradient,
                         file_directory="./data/pickled_data"):

    # write data to output files
    os.makedirs(file_directory,
                exist_ok=True)

    with open(os.path.join(file_directory, "hyperparameters.p"), "wb") as pickle_file:
        pickle.dump(network_hyperparameters, pickle_file)

    with open(os.path.join(file_directory, "input_weights.p"), "wb") as pickle_file:
        pickle.dump(input_weights, pickle_file)

    with open(os.path.join(file_directory, "recurrent_weights.p"), "wb") as pickle_file:
        pickle.dump(recurrent_weights, pickle_file)

    with open(os.path.join(file_directory, "membrane_time_constants.p"), "wb") as pickle_file:
        pickle.dump(membrane_time_constants, pickle_file)

    with open(os.path.join(file_directory, "output_weights.p"), "wb") as pickle_file:
        pickle.dump(output_weights, pickle_file)

    with open(os.path.join(file_directory, "time_series_data.p"), "wb") as pickle_file:
        pickle.dump(time_series_data, pickle_file)

    with open(os.path.join(file_directory, "resulting_voltages.p"), "wb") as pickle_file:
        pickle.dump(resulting_voltages, pickle_file)

    with open(os.path.join(file_directory, "resulting_activations.p"), "wb") as pickle_file:
        pickle.dump(resulting_activations, pickle_file)

    with open(os.path.join(file_directory, "resulting_input_gradient.p"), "wb") as pickle_file:
        pickle.dump(resulting_input_gradient, pickle_file)

    with open(os.path.join(file_directory, "resulting_recurrent_gradient.p"), "wb") as pickle_file:
        pickle.dump(resulting_recurrent_gradient, pickle_file)

    with open(os.path.join(file_directory, "expected_input_gradient.p"), "wb") as pickle_file:
        pickle.dump(expected_input_gradient, pickle_file)

    with open(os.path.join(file_directory, "expected_recurrent_gradient.p"), "wb") as pickle_file:
        pickle.dump(expected_recurrent_gradient, pickle_file)

    print("\nSuccessfully saved to pickle files!")