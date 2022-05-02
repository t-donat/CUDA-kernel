import numpy as np


def verify_forward_pass(expected_voltages, expected_activations, calculated_voltages, calculated_activations):

    cuda_voltages_ok = all([np.allclose(time_step_data, time_step_data[0]) for time_step_data in calculated_voltages])
    cuda_activities_ok = all([np.allclose(time_step_data, time_step_data[0]) for time_step_data in calculated_activations])

    python_voltages_ok = np.allclose(expected_voltages, expected_voltages[:, 0, None])
    python_activations_ok = np.allclose(expected_activations, expected_activations[:, 0, None])

    voltages_match = np.allclose(expected_voltages, calculated_voltages)
    activities_match = np.allclose(expected_activations, calculated_activations)

    return cuda_voltages_ok, cuda_activities_ok, python_voltages_ok, python_activations_ok, voltages_match, activities_match


def print_voltage_discrepancies(expected_voltages, calculated_voltages):

    # epsilon = 1e-5  # For numerical stability

    print("\nMembrane voltage discrepancies (larger than 10⁻⁸):\n")

    num_time_steps, num_batches, num_neurons = expected_voltages.shape

    absolute_differences = np.abs(expected_voltages - calculated_voltages)
    different_spots = np.invert(np.isclose(expected_voltages, calculated_voltages))

    discrepancy_values = absolute_differences[different_spots]
    # to avoid repeating the same difference that is shared across the batches
    #duplicates_removed = discrepancy_values[::num_batches]

    orders_of_magnitude = np.floor(np.log10(np.abs(discrepancy_values))).astype(int)
    unique_orders_of_magnitude, magnitude_counts = np.unique(orders_of_magnitude, return_counts=True)

    num_discrepancies = np.sum(different_spots)
    percentage_discrepancies = np.mean(different_spots)

    print('Order of Magnitude \tNumber of samples \tPercentage of discrepancies')
    for current_magnitude, current_counts in zip(unique_orders_of_magnitude, magnitude_counts):
        print(f"{current_magnitude} \t\t\t{current_counts} \t\t\t{round(current_counts / num_discrepancies * 100, 2)}%")

    print(
        f"\nTotal number of discrepancies: {num_discrepancies} ({round(percentage_discrepancies * 100, 2)}% of all values)")


def print_input_weight_discrepancies(expected_weights, calculated_weights):

    epsilon = 1e-6  # For numerical stability

    print("\nInput weights RELATIVE discrepancies:\n")

    absolute_differences = np.abs(calculated_weights - expected_weights)
    relative_differences = np.abs(absolute_differences / (expected_weights + epsilon))
    different_spots = np.invert(np.isclose(expected_weights, calculated_weights))
    discrepancy_values = relative_differences[different_spots]

    orders_of_magnitude = np.floor(np.log10(discrepancy_values)).astype(int)
    unique_orders_of_magnitude, magnitude_counts = np.unique(orders_of_magnitude, return_counts=True)

    num_discrepancies = discrepancy_values.size

    print('Order of Magnitude \tNumber of samples \tPercentage of discrepancies')
    for current_magnitude, current_counts in zip(unique_orders_of_magnitude, magnitude_counts):
        print(f"{current_magnitude} \t\t\t{current_counts} \t\t\t{round(current_counts / num_discrepancies * 100, 2)}%")

    order_of_magnitude_weights = np.floor(np.log10(np.abs(expected_weights) + epsilon)).astype(int)

    print(f"\nMedian order of magnitude of the weights: {np.median(order_of_magnitude_weights)}")
    print(f"Total number of discrepancies: {num_discrepancies} ({round(num_discrepancies / expected_weights.size * 100, 2)}% of all values)")


def print_weight_discrepancies(name_of_weights, expected_weights, calculated_weights):

    epsilon = 1e-6  # For numerical stability

    print(f"\n{name_of_weights} weights RELATIVE discrepancies:\n")

    absolute_differences = np.abs(calculated_weights - expected_weights)
    relative_differences = np.abs(absolute_differences / (expected_weights + epsilon))
    different_spots = np.invert(np.isclose(expected_weights, calculated_weights))
    discrepancy_values = relative_differences[different_spots]

    orders_of_magnitude = np.floor(np.log10(discrepancy_values)).astype(int)
    unique_orders_of_magnitude, magnitude_counts = np.unique(orders_of_magnitude, return_counts=True)

    num_discrepancies = discrepancy_values.size

    print('Order of Magnitude \tNumber of samples \tPercentage of discrepancies')
    for current_magnitude, current_counts in zip(unique_orders_of_magnitude, magnitude_counts):
        print(f"{current_magnitude} \t\t\t{current_counts} \t\t\t{round(current_counts / num_discrepancies * 100, 2)}%")

    order_of_magnitude_weights = np.floor(np.log10(np.abs(expected_weights) + epsilon)).astype(int)

    print(f"\nMedian order of magnitude of the weights: {np.median(order_of_magnitude_weights)}")
    print(f"Total number of discrepancies: {num_discrepancies} ({round(num_discrepancies / expected_weights.size * 100, 2)}% of all values)")


def print_membrane_decay_weight_discepancies(expected_dE_dalpha, resulting_dE_dalpha):

    print("\nMembrane decay weights RELATIVE discrepancies:\n")


def verify_backward_pass(expected_input_gradient, expected_recurrent_gradient, expected_time_constant_gradient,
                         calculated_input_gradient, calculated_recurrent_gradient, calculated_time_constant_gradient):

    input_weights_match = np.allclose(expected_input_gradient, calculated_input_gradient)
    recurrent_weights_match = np.allclose(expected_recurrent_gradient, calculated_recurrent_gradient)
    membrane_time_constants_match = np.allclose(expected_time_constant_gradient, calculated_time_constant_gradient)

    return input_weights_match, recurrent_weights_match, membrane_time_constants_match