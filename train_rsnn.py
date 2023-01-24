import os
import argparse
from rsnn_utils.classifier import SpikingNeuralNetworkClassifier, DataLoader

data_set_path = "../BCI_Data/Data/dataset/B_32"
output_path = "."

shared_library_path = "./spiking_network.so"
weights_save_path = os.path.join(output_path, "rsnn_model.hdf5")

num_neurons = 128

# Set up the network
data_set = DataLoader(data_set_path)
classifier = SpikingNeuralNetworkClassifier(data_set_path, num_neurons,
                                            cuda_source_file=shared_library_path,
                                            quiet_mode=False)

classifier.hp.define(initial_membrane_time_constant=200/1000,
                     output_time_window=100,
                     threshold_voltage=1,
                     expected_firing_rate=0.2,
                     gradient_scaling_factor=0.3,
                     learning_rate=0.001,
                     weight_decay_rate=0.0001,
                     firing_rate_lambda=1000.0,
                     time_constant_lambda=1.0,
                     network_dropout_rate=0.3,
                     input_dropout_rate=0.3,
                     gradient_clipping_value=1_000,
                     train_time_constants=False)

classifier.randomly_initialize_weights()
classifier.set_up()

# Train the network
classifier.train(data_set, num_epochs=10)

# Run on test set
test_accuracy, test_loss = classifier.evaluate(data_set.test_samples, data_set.test_labels)

rounded_test_loss = round(float(test_loss), 3)
rounded_test_accuracy = round(float(test_accuracy * 100), 2)

print("\nEvaluation:")
print(f"Test: Loss: {rounded_test_loss}, Accuracy: {rounded_test_accuracy}%")

print("\nUse the best performing parameters from early stopping:")
classifier.use_best_performing_parameters()

test_accuracy, test_loss = classifier.evaluate(data_set.test_samples, data_set.test_labels)

rounded_test_loss = round(float(test_loss), 3)
rounded_test_accuracy = round(float(test_accuracy * 100), 2)

print("\nEvaluation:")
print(f"Test: Loss: {rounded_test_loss}, Accuracy: {rounded_test_accuracy}%")

print(f"\nSaving weights to {weights_save_path}")
classifier.save_to_hdf5(weights_save_path)

print("Done")
