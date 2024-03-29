import os
import numpy as np
import pickle
import json
import h5py
# import argparse
import time
# import copy
# import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Any, Union, Optional

from rsnn_utils.rsnn import randomly_generate_weights, calculate_spike_gradient, convert_batch_to_tensors
from rsnn_utils.data import find_indices_of_max_probabilities
from rsnn_utils.utils import check_type, convert_seconds_to_dhms


# arguments = parser.parse_args()
# input_directory = arguments.input_directory
# output_directory = arguments.output_directory
# job_id = arguments.job_id
# num_epochs = int(arguments.num_epochs)
# num_neurons = int(arguments.num_neurons)
# num_trials = int(arguments.num_trials)


class DataLoader:
    """TODO: Documentation"""

    def __init__(self, input_directory: str):
        self.input_directory = input_directory

        (use_validation_set, train_batch_sizes,
         validation_batch_sizes, test_batch_sizes) = self._load_data_sets()

        self.has_validation_set = use_validation_set

        self.train_batch_sizes = train_batch_sizes
        self.validation_batch_sizes = validation_batch_sizes
        self.test_batch_sizes = test_batch_sizes

    @property
    def input_directory(self):
        return self._input_directory

    @input_directory.setter
    def input_directory(self, new_input_directory):
        check_type(new_input_directory, str, "input_directory")
        self._input_directory = new_input_directory

    @property
    def has_validation_set(self):
        return self._has_validation_set

    @has_validation_set.setter
    def has_validation_set(self, new_has_validation_set):
        check_type(new_has_validation_set, bool, "has_validation_set")
        self._has_validation_set = new_has_validation_set

    @property
    def train_batch_sizes(self):
        return self._train_batch_sizes

    @train_batch_sizes.setter
    def train_batch_sizes(self, new_train_batch_sizes):
        check_type(new_train_batch_sizes, list, "train_batch_sizes")
        self._train_batch_sizes = new_train_batch_sizes

    @property
    def validation_batch_sizes(self):
        return self._validation_batch_sizes

    @validation_batch_sizes.setter
    def validation_batch_sizes(self, new_validation_batch_sizes):
        check_type(new_validation_batch_sizes, list, "validation_batch_sizes")
        self._validation_batch_sizes = new_validation_batch_sizes

    @property
    def test_batch_sizes(self):
        return self._test_batch_sizes

    @test_batch_sizes.setter
    def test_batch_sizes(self, new_test_batch_sizes):
        check_type(new_test_batch_sizes, list, "test_batch_sizes")
        self._test_batch_sizes = new_test_batch_sizes

    def _load_data_sets(self):
        """TODO: Documentation"""

        # train set
        train_set_file_path = os.path.join(self.input_directory, "train_data_set.p")

        with open(train_set_file_path, "rb") as pickle_file:
            train_data_set = pickle.load(pickle_file)

        train_samples, train_labels = train_data_set
        train_samples_as_tensors, train_batch_sizes = convert_batch_to_tensors(train_samples)
        train_labels_as_tensors, _ = convert_batch_to_tensors(train_labels)

        self.train_samples = train_samples_as_tensors
        self.train_labels = train_labels_as_tensors

        # validation set
        val_set_file_path = os.path.join(self.input_directory, "validation_data_set.p")

        if os.path.exists(val_set_file_path):
            print("Found validation set")
            use_validation_set = True

            with open(val_set_file_path, "rb") as pickle_file:
                validation_data_set = pickle.load(pickle_file)

            validation_samples, validation_labels = validation_data_set
            validation_samples_as_tensors, validation_batch_sizes = convert_batch_to_tensors(validation_samples)
            validation_labels_as_tensors, _ = convert_batch_to_tensors(validation_labels)

        else:
            use_validation_set = False

            validation_samples_as_tensors = None
            validation_labels_as_tensors = None
            validation_batch_sizes = 0

        self.validation_samples = validation_samples_as_tensors
        self.validation_labels = validation_labels_as_tensors

        # test set
        test_set_file_path = os.path.join(self.input_directory, "test_data_set.p")

        with open(test_set_file_path, "rb") as pickle_file:
            test_data_set = pickle.load(pickle_file)

        test_samples, test_labels = test_data_set

        test_samples_as_tensors, test_batch_sizes = convert_batch_to_tensors(test_samples)
        test_labels_as_tensors, test_batch_sizes = convert_batch_to_tensors(test_labels)

        self.test_samples = test_samples_as_tensors
        self.test_labels = test_labels_as_tensors

        return (use_validation_set, train_batch_sizes,
                validation_batch_sizes, test_batch_sizes)


class Hyperparameters:
    """TODO: Documentation"""

    def __init__(self, input_directory: str):
        self.input_directory = input_directory
        sampling_frequency, num_classes, num_time_steps, num_input_channels = self._load_base_hyperparameters()

        self.sampling_frequency = sampling_frequency

        self.num_neurons = None
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps
        self.num_input_channels = num_input_channels

        self.initial_membrane_time_constant = None

        self.output_time_window = None
        self.threshold_voltage = None
        self.expected_firing_rate = None

        self.gradient_scaling_factor = None

        self.learning_rate = None
        self.weight_decay_rate = None
        self.firing_rate_lambda = None
        self.time_constant_lambda = None

        self.network_dropout_rate = None
        self.input_dropout_rate = None
        self.gradient_clipping_value = None

        self.train_time_constants = None

    @property
    def dt(self) -> float:
        return 1 / self.sampling_frequency

    @dt.setter
    def dt(self, new_dt) -> None:
        # always set the sampling frequency, even if a fixed dt is specified
        self.sampling_frequency = 1/new_dt

    @property
    def hyperparameter_list(self):
        return [attribute for attribute in self.__dir__()
                if not attribute.startswith("_")
                and attribute not in ["complete",
                                      "load_from_json",
                                      "save_to_json",
                                      "define",
                                      "input_directory",
                                      "hyperparameter_list",
                                      "to_dict"]]

    @property
    def complete(self):
        """TODO: Documentation"""
        public_attributes = self.hyperparameter_list

        all_hyperparameters_defined = all([getattr(self, attribute) is not None for attribute in public_attributes])

        return all_hyperparameters_defined

    def _load_base_hyperparameters(self):
        """TODO: Documentation"""
        # existing hyperparameters
        hyperparameter_file_path = os.path.join(self.input_directory, "hyperparameters.json")

        with open(hyperparameter_file_path, "r") as file:
            original_hyperparameters = json.load(file)

        sampling_frequency = original_hyperparameters["sampling_frequency"]
        num_classes = original_hyperparameters["num_classes"]
        num_time_steps = original_hyperparameters["num_time_steps"]
        num_input_channels = original_hyperparameters["num_input_channels"]

        return sampling_frequency, num_classes, num_time_steps, num_input_channels

    def load_from_json(self, file_name: Optional[str] = None):
        """Loads hyperparameter values from a file on disk

        TODO: Finish Documentation

        """

        check_type(file_name, Optional[str], "file_name")

        if file_name is None:
            file_name = "hyperparameters.json"

        with open(file_name, "r") as json_file:
            hyperparameters = json.load(json_file)

        self.define(**hyperparameters)

    def to_dict(self) -> dict:
        return {hyperparameter_name: getattr(self, hyperparameter_name)
                for hyperparameter_name in self.hyperparameter_list}

    def save_to_json(self, file_name: Optional[str] = None):
        """Creates a dict of hyperparameter names and values and saves it to disk

        TODO: Finish Documentation

        """

        check_type(file_name, Optional[str], "file_name")

        if file_name is None:
            file_name = "hyperparameters.json"

        hyperparameters = self.to_dict()

        with open(file_name, "w") as json_file:
            json.dump(hyperparameters, json_file, indent=4)

    def define(self, **kwargs):
        """TODO: Documentation"""

        for hyperparameter_name, hyperparameter_value in kwargs.items():

            if hasattr(self, hyperparameter_name):
                setattr(self, hyperparameter_name, hyperparameter_value)

            else:
                raise ValueError(f"Unknown hyperparameter '{hyperparameter_name}'")


class ModelHistory:
    """TODO: Documentation"""

    def __init__(self, debug_mode: bool = False):

        self.training_stats = {"accuracy": [],
                               "overall_loss": [],
                               "cross_entropy_loss": [],
                               "fire_rate_loss": [],
                               "time_constant_loss": [],
                               "break_due_to_nan": False,
                               "training_duration": ""}

        self.epoch_stats = {"accuracy": [],
                            "overall_loss": [],
                            "cross_entropy_loss": [],
                            "fire_rate_loss": [],
                            "time_constant_loss": []}

        self.validation_stats = {"accuracy": [],
                                 "loss": []}

        self.best_performing_parameters = {"val_accuracy": 0.0,
                                           "W_in": None,
                                           "W_rec": None,
                                           "W_out": None,
                                           "tau_membrane": None}

        if debug_mode:

            time_constant_derivatives_over_epochs = []

            w_in_stats = {"mean": [],
                          "std": [],
                          "norm": []}

            w_rec_stats = {"mean": [],
                           "std": [],
                           "norm": []}

            w_out_stats = {"mean": [],
                           "std": [],
                           "norm": []}

            w_in_derivative_stats = {"mean": [],
                                     "std": [],
                                     "norm": [],
                                     "clipped_norm": []}

            w_rec_derivative_stats = {"mean": [],
                                      "std": [],
                                      "norm": [],
                                      "clipped_norm": []}

            w_out_derivative_stats = {"mean": [],
                                      "std": [],
                                      "norm": [],
                                      "clipped_norm": []}

            tau_derivative_stats = {"mean": [],
                                    "std": [],
                                    "norm": [],
                                    "clipped_norm": []}

            overshooting_batches = {"epoch": [],
                                    "batch": [],
                                    "w_in": [],
                                    "w_rec": [],
                                    "w_out": [],
                                    "tau_membrane": [],
                                    "w_in_derivative": [],
                                    "w_rec_derivative": [],
                                    "w_out_derivative": [],
                                    "tau_membrane_derivative": []}

            self.debug_stats = {"time_constant_derivatives_over_epochs": time_constant_derivatives_over_epochs,
                                "w_in_stats": w_in_stats,
                                "w_rec_stats": w_rec_stats,
                                "w_out_stats": w_out_stats,
                                "w_in_derivative_stats": w_in_derivative_stats,
                                "w_rec_derivative_stats": w_rec_derivative_stats,
                                "w_out_derivative_stats": w_out_derivative_stats,
                                "tau_derivative_stats": tau_derivative_stats,
                                "overshooting_batches": overshooting_batches}

        else:
            self.debug_stats = {}

    def summarize_epoch(self, batch_sizes=None):
        """Average over the batch-wise accuracies and losses to get the corresponding values for the entire epoch

        Optionally this operation can be expanded by the sizes of each batch in the epoch using the 'batch_sizes'
        argument. This can be done in order to take different batch sizes (number of samples per batch) into account
        when calculating the metrics for the entire epoch. Otherwise, all epochs are assumed to have the same number of
        samples.

        """

        num_batches = len(self.epoch_stats["accuracy"])

        if batch_sizes is None:
            batch_sizes = np.ones(num_batches)

        batch_sizes = np.asarray(batch_sizes)

        num_specified_batch_sizes = len(batch_sizes)

        if num_batches != num_specified_batch_sizes:
            error_message_part_1 = f"Expected batch sizes for {num_batches} many batches, but only "
            error_message_part_2 = f"{num_specified_batch_sizes} many were specified"
            full_error_message = error_message_part_1 + error_message_part_2

            raise ValueError(full_error_message)

        self._summarize_metric("accuracy", batch_sizes)
        self._summarize_metric("overall_loss", batch_sizes)
        self._summarize_metric("cross_entropy_loss", batch_sizes)
        self._summarize_metric("fire_rate_loss", batch_sizes)
        self._summarize_metric("time_constant_loss", batch_sizes)

        # Manually written out:
        # accuracies_in_epoch = np.asarray(self.epoch_stats["accuracy"])
        # epoch_accuracy = np.mean(accuracies_in_epoch * batch_sizes)
        # self.training_stats["accuracy"].append(epoch_accuracy)
        #
        # overall_loss_in_epoch = np.asarray(self.epoch_stats["overall_loss"])
        # epoch_overall_loss = np.mean(overall_loss_in_epoch * batch_sizes)
        # self.training_stats["overall_loss"].append(epoch_overall_loss)
        #
        # cross_entropy_loss_in_epoch = np.asarray(self.epoch_stats["cross_entropy_loss"])
        # epoch_cross_entropy_loss = np.mean(cross_entropy_loss_in_epoch * batch_sizes)
        # self.training_stats["cross_entropy_loss"].append(epoch_cross_entropy_loss)
        #
        # fire_rate_loss_in_epoch = np.asarray(self.epoch_stats["fire_rate_loss"])
        # epoch_fire_rate_loss = np.mean(fire_rate_loss_in_epoch * batch_sizes)
        # self.training_stats["fire_rate_loss"].append(epoch_fire_rate_loss)
        #
        # time_constant_loss_in_epoch = np.asarray(self.epoch_stats["time_constant_loss"])
        # epoch_time_constant_loss = np.mean(time_constant_loss_in_epoch * batch_sizes)
        # self.training_stats["time_constant_loss"].append(epoch_time_constant_loss)

    def _summarize_metric(self, metric_name: str, batch_sizes: np.ndarray) -> None:
        """TODO: Documentation"""

        if metric_name not in self.training_stats:
            raise ValueError(f"Unknown metric '{metric_name}'")

        else:
            metric_value_per_batch = np.asarray(self.epoch_stats[metric_name])
            metric_value_over_epoch = np.sum(metric_value_per_batch * batch_sizes)/np.sum(batch_sizes)
            self.training_stats[metric_name].append(metric_value_over_epoch)

    def start_epoch(self):

        # reset metric lists
        self.epoch_stats["accuracy"] = []
        self.epoch_stats["overall_loss"] = []
        self.epoch_stats["cross_entropy_loss"] = []
        self.epoch_stats["fire_rate_loss"] = []
        self.epoch_stats["time_constant_loss"] = []


class SpikingNeuralNetworkClassifier:
    """TODO: Documentation"""

    def __init__(self, input_directory: str, num_neurons: int,
                 cuda_source_file: Optional[str] = None,
                 quiet_mode: bool = True, verbose_mode: bool = False, debug_mode: bool = False):

        # set up
        self.input_directory = input_directory

        # self.num_neurons = num_neurons

        # self.data_set = DataLoader(input_directory)
        self.hp = Hyperparameters(self.input_directory)
        self.hp.define(num_neurons=num_neurons)

        self.model_history = ModelHistory(debug_mode=debug_mode)

        self.cuda_source_library = self._load_cuda_source_library(cuda_source_file)

        self.W_in = None
        self.W_rec = None
        self.W_out = None
        self.tau_membrane = None

        self.optimizer = None

        self.quiet_mode = quiet_mode
        self.verbose_mode = verbose_mode
        self.debug_mode = debug_mode

        self.break_due_to_nan = False

        self.cache = {"dropout_corrected_activations": None,
                      "network_output": None,
                      "smoothed_spikes": None,
                      "resulting_voltages": None,
                      "dropout_corrected_batch_data": None,
                      "predicted_distribution": None,
                      "actual_firing_rates": None}

    @property
    def parameters(self):
        """TODO: Documentation"""

        if self.hp.train_time_constants is None:
            raise ValueError("Hyperparameter 'train_time_constants' is still undefined!")

        elif self.hp.train_time_constants:
            return [self.W_in, self.W_rec, self.W_out, self.tau_membrane]

        else:
            return [self.W_in, self.W_rec, self.W_out]

    def set_up(self):
        """TODO: Documentation"""

        if not self.hp.complete:
            raise ValueError("Not all hyperparameters have been defined yet!")

        self.optimizer = tfa.optimizers.AdamW(learning_rate=self.hp.learning_rate,
                                              weight_decay=self.hp.weight_decay_rate)

    def train(self, data_set: DataLoader, num_epochs: int):
        """TODO: Documentation, Debug Mode"""

        debug_interval = int(np.ceil(num_epochs / 10))
        self.break_due_to_nan = False

        start = time.time()

        for current_epoch in range(1, num_epochs + 1):

            batch_sizes = []
            self.model_history.start_epoch()

            for batch_number, (batch_data, batch_labels) in enumerate(zip(data_set.train_samples,
                                                                          data_set.train_labels)):
                current_batch_size = batch_data.shape[1]
                batch_sizes.append(current_batch_size)

                # Forward Pass
                softmax_output = self._forward_pass(batch_data, current_batch_size)

                # Evaluation
                time_step_with_highest_prob_per_sample, predicted_classes = self._classify(softmax_output)
                self._calculate_loss(softmax_output, time_step_with_highest_prob_per_sample, batch_labels)
                self._calculate_accuracy(predicted_classes, batch_labels)

                # Backward Pass
                (dE_dW_in, dE_dW_rec,
                 dE_dtau_membrane, dE_dW_out) = self._backward_pass(batch_labels,
                                                                    time_step_with_highest_prob_per_sample,
                                                                    current_batch_size)

                # Weight Update
                self._update_weights(dE_dW_in, dE_dW_rec, dE_dtau_membrane, dE_dW_out)

                if self.break_due_to_nan:
                    self._write_to_stdout(f"Loss turned into NaN at epoch {current_epoch}, batch {batch_number}")
                    break

            self.model_history.summarize_epoch(batch_sizes)

            if data_set.has_validation_set:
                self._evaluate_on_validation_set(data_set.validation_samples, data_set.validation_labels)

            if not self.quiet_mode and (current_epoch == 1 or current_epoch % debug_interval == 0):

                self._update_message(current_epoch,
                                     self.model_history.training_stats,
                                     self.model_history.validation_stats)

            if self.break_due_to_nan:
                break

        training_duration = time.time() - start
        duration_time_string = convert_seconds_to_dhms(training_duration)

        self.model_history.training_stats["break_due_to_nan"] = self.break_due_to_nan
        self.model_history.training_stats["training_duration"] = duration_time_string

    def predict(self, input_data):
        """TODO: Implement"""

        predictions = []

        for batch_data in input_data:
            current_batch_size = batch_data.shape[1]

            softmax_output = self._forward_pass(batch_data, current_batch_size)

            # EVALUATION
            _, predicted_classes = self._classify(softmax_output)

            predictions.append(predicted_classes)

        predictions = tf.convert_to_tensor(predictions, dtype=tf.int32)

        return predictions

    def evaluate(self, input_data, input_labels):
        """TODO: Implement"""

        model_predictions = []
        ground_truth_labels = []
        batch_losses = []

        for batch_data, batch_labels in zip(input_data, input_labels):
            current_batch_size = batch_data.shape[1]

            # FORWARD PASS
            softmax_output = self._forward_pass(batch_data, current_batch_size)

            # EVALUATION
            time_step_with_highest_prob_per_sample, predicted_classes = self._classify(softmax_output)

            # Loss
            (overall_loss, cross_entropy_loss,
             firing_rate_loss, time_constant_loss) = self._calculate_loss(softmax_output,
                                                                          time_step_with_highest_prob_per_sample,
                                                                          batch_labels)

            actual_classes = tf.where(batch_labels)[:, 1]

            model_predictions.append(predicted_classes)
            ground_truth_labels.append(actual_classes)
            batch_losses.append(overall_loss * current_batch_size)

        model_predictions = tf.convert_to_tensor(model_predictions, dtype=tf.int64)
        model_predictions = tf.reshape(model_predictions, [-1])
        ground_truth_labels = tf.reshape(ground_truth_labels, [-1])

        accuracy = tf.math.reduce_mean(tf.cast(model_predictions == ground_truth_labels, dtype=tf.float32)).numpy()
        loss = tf.math.reduce_mean(batch_losses).numpy()

        return accuracy, loss

    def _evaluate_on_validation_set(self, validation_samples, validation_labels):
        """TODO: Documentation"""

        validation_set_accuracy, validation_set_loss = self.evaluate(validation_samples, validation_labels)

        self.model_history.validation_stats["loss"].append(validation_set_loss)
        self.model_history.validation_stats["accuracy"].append(validation_set_accuracy)

        if validation_set_accuracy > self.model_history.best_performing_parameters["val_accuracy"]:
            self.model_history.best_performing_parameters["val_accuracy"] = validation_set_accuracy
            self.model_history.best_performing_parameters["W_in"] = tf.identity(self.W_in)
            self.model_history.best_performing_parameters["W_rec"] = tf.identity(self.W_rec)
            self.model_history.best_performing_parameters["W_out"] = tf.identity(self.W_out)
            self.model_history.best_performing_parameters["tau_membrane"] = tf.identity(self.tau_membrane)

    @staticmethod
    def _load_cuda_source_library(cuda_source_file: Optional[str] = None):
        """Load the shared library containing the CUDA operations of the RSNN

        If left as None, a shared library with the file name 'spiking_network.so' is looked for in the current working
        directory

        Args:
            cuda_source_file: (Optional[str]) The file path to the shared library. Default is None

        Returns:
            A module containing the forward and backward pass operations of the RSNN

        Raises:
            TypeError: If 'cuda_source_file' is not a str or None
            FileNotFoundError: If no file exists at the path specified in 'cuda_source_file'

        """

        check_type(cuda_source_file, Optional[str], "cuda_source_file")

        if cuda_source_file is None:
            cuda_source_file = os.path.join(os.path.dirname(__file__), "spiking_network.so")

        if not os.path.exists(cuda_source_file):
            raise FileNotFoundError(f"Could not find shared library at expected path: {cuda_source_file}")

        source_library = tf.load_op_library(cuda_source_file)

        if "forward_pass" not in dir(source_library):
            raise ValueError(f"Could not find 'forward_pass' op in the shared library at {cuda_source_file}")

        elif "backward_pass" not in dir(source_library):
            raise ValueError(f"Could not find 'backward_pass' op in the shared library at {cuda_source_file}")

        return source_library

    def randomly_initialize_weights(self):
        (W_in_init, W_rec_init,
         W_out_init, tau_membrane_init) = randomly_generate_weights(self.hp.num_neurons,
                                                                    self.hp.num_input_channels,
                                                                    self.hp.num_classes,
                                                                    self.hp.threshold_voltage,
                                                                    self.hp.initial_membrane_time_constant)

        self._initialize_weights(W_in_init, W_rec_init, W_out_init, tau_membrane_init)

    def load_from_hdf5(self, hdf5_file_path: str) -> None:

        if not os.path.exists(hdf5_file_path):
            raise FileNotFoundError(f"Could not find an hd5f file at {hdf5_file_path}")

        with h5py.File(hdf5_file_path, "r") as file:

            # model weights
            if "weights" not in file:
                raise ValueError("Group 'weights' missing from the hdf5-file!")

            weights = file["weights"]
            W_in = tf.convert_to_tensor(weights["W_in"][:], dtype=tf.float32)
            W_rec = tf.convert_to_tensor(weights["W_rec"][:], dtype=tf.float32)
            W_out = tf.convert_to_tensor(weights["W_out"][:], dtype=tf.float32)
            tau_membrane = tf.convert_to_tensor(weights["tau_membrane"][:], dtype=tf.float32)

            self._initialize_weights(W_in, W_rec, W_out, tau_membrane)

            # model hyperparameters
            if "hyperparameters" not in file:
                raise ValueError("Group 'hyperparameters' missing from the hdf5-file!")

            loaded_hyperparameters = {hp_name: hp_value[0]
                                      for hp_name, hp_value in file["hyperparameters"].items()}
            self.hp.define(**loaded_hyperparameters)

    def save_to_hdf5(self, hdf5_file_path: str) -> None:

        with h5py.File(hdf5_file_path, "w") as file:

            weights = file.create_group("weights")

            weights.create_dataset("W_in", data=self.W_in)
            weights.create_dataset("W_rec", data=self.W_rec)
            weights.create_dataset("W_out", data=self.W_out)
            weights.create_dataset("tau_membrane", data=self.tau_membrane)

            hyperparameters = file.create_group("hyperparameters")

            model_hyperparameters = self.hp.to_dict()
            for hp_name, hp_value in model_hyperparameters.items():
                hyperparameters.create_dataset(hp_name, data=np.array([hp_value]))

    def _initialize_weights(self, W_in, W_rec, W_out, tau_membrane):
        """TODO: Documentation"""

        with tf.device("/cpu"):
            self.W_in = tf.Variable(W_in, dtype=tf.float32)
            self.W_rec = tf.Variable(W_rec, dtype=tf.float32)
            self.W_out = tf.Variable(W_out, dtype=tf.float32)
            self.tau_membrane = tf.Variable(tau_membrane, dtype=tf.float32)

        # self.W_in = W_in
        # self.W_rec = W_rec
        # self.W_out = W_out
        # self.tau_membrane = tau_membrane

    def _forward_pass(self, batch_data, current_batch_size):
        """TODO: Documentation"""

        # Input Dropout
        random_distribution = tf.random.uniform(shape=batch_data.get_shape())
        will_not_be_dropped = tf.cast(random_distribution > self.hp.input_dropout_rate, dtype=tf.float32)

        correction_factor = 1 - self.hp.input_dropout_rate
        dropout_corrected_batch_data = (batch_data * will_not_be_dropped) / correction_factor

        # Forward Pass
        resulting_voltages, resulting_activations = self.rsnn_forward_pass(self.W_in, self.W_rec, self.tau_membrane,
                                                                           dropout_corrected_batch_data)
        # Network Dropout
        random_distribution = tf.random.uniform(shape=resulting_activations.get_shape())
        will_not_be_dropped = tf.cast(random_distribution > self.hp.network_dropout_rate, dtype=tf.float32)

        correction_factor = 1 - self.hp.network_dropout_rate  # could precompute
        dropout_corrected_activations = (resulting_activations * will_not_be_dropped) / correction_factor

        # Continue with Forward Pass
        smoothed_spikes = tf.stack(
            [tf.math.reduce_mean(dropout_corrected_activations[i - self.hp.output_time_window: i], axis=0)
             if i >= self.hp.output_time_window else tf.zeros(shape=[current_batch_size, self.hp.num_neurons])
             for i in range(1, self.hp.num_time_steps + 1)])

        network_output = tf.linalg.matmul(smoothed_spikes, self.W_out, transpose_b=True)

        softmax_output = tf.math.exp(network_output - tf.math.reduce_max(network_output))
        softmax_output = softmax_output / tf.math.reduce_sum(softmax_output, axis=-1, keepdims=True)

        # Cache for loss and gradient calculation
        self.cache["dropout_corrected_activations"] = dropout_corrected_activations
        self.cache["network_output"] = network_output
        self.cache["smoothed_spikes"] = smoothed_spikes
        self.cache["resulting_voltages"] = resulting_voltages
        self.cache["dropout_corrected_batch_data"] = dropout_corrected_batch_data

        return softmax_output

    def _calculate_loss(self, softmax_output, time_step_with_highest_prob_per_sample, ground_truth_distribution):
        """TODO: Documentation"""

        # Cross Entropy Loss between the predicted distribution and the ground truth distribution
        # Only the time and batch index are needed to calculate this loss
        predicted_distribution = tf.gather_nd(softmax_output, time_step_with_highest_prob_per_sample)

        # 'batch_labels' contains the one hot encoded ground truth
        # ground_truth_distribution = batch_labels

        cross_entropy_loss = tf.math.reduce_mean(-ground_truth_distribution*tf.math.log(predicted_distribution)).numpy()

        # Firing Rate Loss (MSE between actual and expected firing rates)
        dropout_corrected_activations = self.cache["dropout_corrected_activations"]
        actual_firing_rates = tf.math.reduce_mean(dropout_corrected_activations, axis=[0, 1])

        quadratic_difference = (actual_firing_rates - self.hp.expected_firing_rate) ** 2
        firing_rate_loss = self.hp.firing_rate_lambda * tf.math.reduce_mean(quadratic_difference).numpy()

        # Membrane Time Constant Loss (MSE between current and initial value of the membrane time constant)
        quadratic_difference = (self.tau_membrane - self.hp.initial_membrane_time_constant) ** 2
        time_constant_loss = self.hp.time_constant_lambda * tf.math.reduce_mean(quadratic_difference).numpy()

        # if tf.math.is_nan(time_constant_loss):
        #     print(f"NaN in epoch {current_epoch}, batch {batch_number}")

        overall_loss = cross_entropy_loss + firing_rate_loss + time_constant_loss

        self.model_history.epoch_stats["overall_loss"].append(overall_loss)
        self.model_history.epoch_stats["cross_entropy_loss"].append(cross_entropy_loss)
        self.model_history.epoch_stats["fire_rate_loss"].append(firing_rate_loss)
        self.model_history.epoch_stats["time_constant_loss"].append(time_constant_loss)

        # firing_rates_over_epochs.append(actual_firing_rates)

        self.cache["predicted_distribution"] = predicted_distribution
        self.cache["actual_firing_rates"] = actual_firing_rates

        if tf.math.is_nan(overall_loss):
            self.break_due_to_nan = True

        return overall_loss, cross_entropy_loss, firing_rate_loss, time_constant_loss

    def _calculate_accuracy(self, predicted_classes, ground_truth_distribution):
        """TODO: Documentation"""

        ground_truth_classes = tf.where(ground_truth_distribution)[:, 1]
        batch_accuracy = tf.reduce_mean(tf.cast(predicted_classes == ground_truth_classes, tf.float32)).numpy()

        self.model_history.epoch_stats["accuracy"].append(batch_accuracy)

    def rsnn_forward_pass(self, W_in, W_rec, tau_membrane, batch_data):

        return self.cuda_source_library.forward_pass(W_in, W_rec, tau_membrane, batch_data,
                                                     threshold_voltage=self.hp.threshold_voltage,
                                                     delta_t=self.hp.dt)

    def rsnn_backward_pass(self, partial_dE_dv, W_rec, tau_membrane, dropout_corrected_batch_data, resulting_voltages,
                           dropout_corrected_activations):

        return self.cuda_source_library.backward_pass(partial_dE_dv, W_rec, tau_membrane, dropout_corrected_batch_data,
                                                      resulting_voltages, dropout_corrected_activations,
                                                      threshold_voltage=self.hp.threshold_voltage, delta_t=self.hp.dt,
                                                      gradient_scaling_factor=self.hp.gradient_scaling_factor)

    @staticmethod
    def _classify(softmax_output):
        """TODO: Documentation"""
        # Evaluation
        indices_with_highest_probability = find_indices_of_max_probabilities(softmax_output)
        time_step_with_highest_prob_per_sample = indices_with_highest_probability[:, :2]
        predicted_classes = indices_with_highest_probability[:, 2]

        return time_step_with_highest_prob_per_sample, predicted_classes

    def _backward_pass(self, ground_truth_distribution, time_step_with_highest_prob_per_sample, current_batch_size):
        """TODO: Documentation"""

        # load the variables from cache that are needed to calculate the gradients
        predicted_distribution = self.cache["predicted_distribution"]
        network_output = self.cache["network_output"]
        smoothed_spikes = self.cache["smoothed_spikes"]
        actual_firing_rates = self.cache["actual_firing_rates"]
        resulting_voltages = self.cache["resulting_voltages"]
        dropout_corrected_batch_data = self.cache["dropout_corrected_batch_data"]
        dropout_corrected_activations = self.cache["dropout_corrected_activations"]

        # Calculation of the gradients starts here
        dE_dnetwork_output_values = predicted_distribution - ground_truth_distribution
        dE_dnetwork_output = tf.scatter_nd(time_step_with_highest_prob_per_sample,
                                           dE_dnetwork_output_values,
                                           network_output.get_shape())

        # both matrix multiplications are calculated in batches (for each time step)
        dE_dW_out_components = tf.matmul(dE_dnetwork_output, smoothed_spikes, transpose_a=True)
        dE_dW_out = tf.math.reduce_sum(dE_dW_out_components, axis=0)

        dE_dsmoothed_spikes = tf.matmul(dE_dnetwork_output, self.W_out)

        original_dE_dz = tf.stack([tf.reduce_mean(dE_dsmoothed_spikes[i - self.hp.output_time_window: i], axis=0)
                                   if i >= self.hp.output_time_window
                                   else tf.zeros(shape=[current_batch_size, self.hp.num_neurons])
                                   for i in range(1, self.hp.num_time_steps + 1)])

        firing_rate_dE_dz_values = actual_firing_rates - self.hp.expected_firing_rate
        firing_rate_dE_dz_values = self.hp.firing_rate_lambda / self.hp.num_neurons * firing_rate_dE_dz_values
        dE_dactual_firing_rate = tf.ones(shape=[self.hp.num_time_steps,
                                                current_batch_size,
                                                self.hp.num_neurons]) * firing_rate_dE_dz_values

        dactual_firing_rate_dz = 1 / (self.hp.num_time_steps * current_batch_size)

        firing_rate_dE_dz = dE_dactual_firing_rate * dactual_firing_rate_dz
        dE_dz = original_dE_dz + firing_rate_dE_dz

        partial_dE_dv = dE_dz * calculate_spike_gradient(resulting_voltages, self.hp.threshold_voltage)

        (dE_dW_in,
         dE_dW_rec,
         original_dE_dtau_membrane) = self.rsnn_backward_pass(partial_dE_dv, self.W_rec, self.tau_membrane,
                                                              dropout_corrected_batch_data, resulting_voltages,
                                                              dropout_corrected_activations)

        membrane_time_constant_dE_dz = self.tau_membrane - self.hp.initial_membrane_time_constant
        membrane_time_constant_dE_dz = self.hp.time_constant_lambda / self.hp.num_neurons * membrane_time_constant_dE_dz

        dE_dtau_membrane = original_dE_dtau_membrane + membrane_time_constant_dE_dz

        return dE_dW_in, dE_dW_rec, dE_dtau_membrane, dE_dW_out

    def _update_weights(self, dE_dW_in, dE_dW_rec, dE_dtau_membrane, dE_dW_out):

        """TODO: Implement"""
        if self.hp.train_time_constants:
            gradients = [dE_dW_in, dE_dW_rec, dE_dW_out, dE_dtau_membrane]
        else:
            gradients = [dE_dW_in, dE_dW_rec, dE_dW_out]

        clipped_gradients = [tf.clip_by_norm(g, self.hp.gradient_clipping_value) for g in gradients]
        global_norm = tf.linalg.global_norm(gradients)

        # clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, gradient_clipping_value)
        global_norm = global_norm.numpy()

        self.optimizer.apply_gradients(zip(clipped_gradients, self.parameters))

        # time_constants_over_epochs.append(tau_membrane.numpy().flatten())
        # all_global_norms.append(global_norm)

    def _update_message(self, current_epoch: int, train_set_metrics: dict, validation_set_metrics: dict) -> None:
        """TODO: Documentation, Debug Mode"""

        train_set_loss = train_set_metrics["overall_loss"][-1]
        train_set_accuracy = train_set_metrics["accuracy"][-1]

        rounded_train_loss = round(float(train_set_loss), 3)
        rounded_train_accuracy = round(float(train_set_accuracy * 100), 2)
        self._write_to_stdout(f"\nEpoch {current_epoch}:")
        self._write_to_stdout(f"Train: Loss: {rounded_train_loss}, Accuracy: {rounded_train_accuracy}%")

        if validation_set_metrics["loss"] and validation_set_metrics["accuracy"]:
            validation_set_loss = validation_set_metrics["loss"][-1]
            validation_set_accuracy = validation_set_metrics["accuracy"][-1]

            rounded_validation_loss = round(float(validation_set_loss), 3)
            rounded_validation_accuracy = round(float(validation_set_accuracy * 100), 2)

            message = f"Validation: Loss: {rounded_validation_loss}, Accuracy: {rounded_validation_accuracy}%"

            self._write_to_stdout(message)

        if self.verbose_mode:
            cross_entropy_loss = train_set_metrics["cross_entropy_loss"][-1]
            fire_rate_loss = train_set_metrics["fire_rate_loss"][-1]
            time_constant_loss = train_set_metrics["time_constant_loss"][-1]

            self._write_to_stdout("\nLosses:")
            self._write_to_stdout(f"Cross entropy loss: {cross_entropy_loss}")
            self._write_to_stdout(f"Fire rate loss: {fire_rate_loss}")
            self._write_to_stdout(f"Time constant loss: {time_constant_loss}")

        # if self.debug_mode:
        #     print("\nDerivatives:")
        #     print(f"Global norm: {global_norm}")
        #     print(f"W_in norm: {w_in_derivative_norm}")
        #     print(f"W_rec norm: {w_rec_derivative_norm}")
        #     print(f"W_out norm: {w_out_derivative_norm}")
        #     print(f"Tau norm: {tau_derivative_norm}")

    def _write_to_stdout(self, message: str) -> None:
        """TODO: Documentation"""

        if not self.quiet_mode:
            print(message)

    def use_best_performing_parameters(self):

        if self.model_history.best_performing_parameters["W_in"] is not None:
            self.W_in = tf.identity(self.model_history.best_performing_parameters["W_in"])
            self.W_rec = tf.identity(self.model_history.best_performing_parameters["W_rec"])
            self.W_out = tf.identity(self.model_history.best_performing_parameters["W_out"])
            self.tau_membrane = tf.identity(self.model_history.best_performing_parameters["tau_membrane"])

        else:
            raise ValueError("The network must be trained first!")
