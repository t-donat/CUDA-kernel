import os
import argparse

import numpy as np
import pandas as pd

from rsnn_utils.run_time import time_python_implementation, time_cuda_implementation
from rsnn_utils.classifier import SpikingNeuralNetworkClassifier, DataLoader


shared_library_path = "./spiking_network.so"

parser = argparse.ArgumentParser(description='Collect run time data of repeated operation runs')
parser.add_argument("-i", "--input_dir",
                    dest="input_directory",
                    help="Directory to search for the data set")

parser.add_argument('-r', '--num_repetitions',
                    type=int, default=100,
                    dest="num_repetitions",
                    help="How many times the operation is rerun for each network size are contained in a batch")

parser.add_argument("-n", "--network_sizes",
                    nargs="+",
                    default=[8, 16, 32, 64, 128],
                    dest="network_sizes",
                    type=int,
                    help="The different network sizes (numbers of neurons) to run the operations on")

parser.add_argument("-o", "--output_dir",
                    dest="output_directory",
                    help="Directory to save the results to")

args = parser.parse_args()
input_directory = args.input_directory
num_repetitions = args.num_repetitions
network_sizes = args.network_sizes
output_directory = args.output_directory

# data_set_path = "../BCI_Data/Data/dataset/B_32"

# -------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------

# defined once for all networks that are tested

initial_membrane_time_constant = 100/1000
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

hyperparameters = {"initial_membrane_time_constant": initial_membrane_time_constant,
                   "output_time_window": output_time_window,
                   "threshold_voltage": threshold_voltage,
                   "expected_firing_rate": expected_firing_rate,
                   "gradient_scaling_factor": gradient_scaling_factor,
                   "learning_rate": learning_rate,
                   "weight_decay_rate": weight_decay_rate,
                   "firing_rate_lambda": firing_rate_lambda,
                   "time_constant_lambda": time_constant_lambda,
                   "network_dropout_rate": network_dropout_rate,
                   "input_dropout_rate": input_dropout_rate,
                   "gradient_clipping_value": gradient_clipping_value,
                   "train_time_constants": train_time_constants}

# -------------------------------------------------------------------
# RUNTIME TEST
# -------------------------------------------------------------------

data_set = DataLoader(input_directory)

sample_batch_for_test = data_set.train_samples[0]
label_batch_for_test = data_set.train_labels[0]

results = {}

for current_num_neurons in network_sizes:

    # Set up RSNN of with 'current_num_neurons' many neurons, initialize weights, etc
    classifier = SpikingNeuralNetworkClassifier(input_directory, output_directory, current_num_neurons,
                                                cuda_source_file=shared_library_path,
                                                quiet_mode=False)

    classifier.hp.define(**hyperparameters)
    classifier.set_up()

    python_forward_pass = []
    python_backward_pass = []
    cuda_forward_pass = []
    cuda_backward_pass = []

    for _ in range(num_repetitions):
        # PYTHON
        python_forward_duration, python_backward_duration = time_python_implementation(sample_batch_for_test.numpy(),
                                                                                       label_batch_for_test.numpy(),
                                                                                       classifier.W_in.numpy(),
                                                                                       classifier.W_rec.numpy(),
                                                                                       classifier.tau_membrane.numpy(),
                                                                                       classifier.W_out.numpy(),
                                                                                       classifier.hp)

        # CUDA
        cuda_forward_duration, cuda_backward_duration = time_cuda_implementation(sample_batch_for_test,
                                                                                 label_batch_for_test,
                                                                                 classifier.W_in,
                                                                                 classifier.W_rec,
                                                                                 classifier.tau_membrane,
                                                                                 classifier.W_out,
                                                                                 classifier.hp,
                                                                                 classifier.cuda_source_library)

        # Save timing results
        python_forward_pass.append(python_forward_duration)
        python_backward_pass.append(python_backward_duration)
        cuda_forward_pass.append(cuda_forward_duration)
        cuda_backward_pass.append(cuda_backward_duration)

    results[current_num_neurons] = {"python": {"forward": python_forward_pass,
                                               "backward": python_backward_pass},
                                    "cuda": {"forward": cuda_forward_pass,
                                             "backward": cuda_backward_pass}}

    # cuda_results[current_num_neurons] =

# -------------------------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------------------------

num_runs = len(results)
run_sizes = [str(current_size) for current_size in results.keys()]
run_sizes_index = np.repeat(run_sizes, 4)

implementations = ["python", "cuda"]
implementations_index = np.repeat(implementations * num_runs, 2)

operations = ["forward", "backward"]
operations_index = np.array(operations * 2 * num_runs)

run_time_indexes = [run_sizes_index,
                    implementations_index,
                    operations_index]

run_times = []

for current_size, current_data in results.items():
    run_times.append(current_data["python"]["forward"])
    run_times.append(current_data["python"]["backward"])

    run_times.append(current_data["cuda"]["forward"])
    run_times.append(current_data["cuda"]["backward"])

run_time_data = np.array(run_times)

results_df = pd.DataFrame(run_time_data, index=run_time_indexes)
results_df = results_df.sort_index()

df_save_path = os.path.join(output_directory, "run_time_results.csv")
results_df.to_csv(df_save_path)

print("Done")
