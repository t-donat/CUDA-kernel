from typing import Any, Union


def check_type(parameter: Any, expected_type: Union[type, tuple[type]], parameter_name: str):

    if not isinstance(parameter, expected_type):

        if isinstance(expected_type, type):
            message_expected_type = str(expected_type)[8:-2]

        elif isinstance(expected_type, tuple):
            message_expected_type = ", ".join([str(element)[8:-2] for element in expected_type])

        else:
            # Union, Optional from typing package
            individual_types = expected_type.__args__
            message_expected_type = ", ".join([str(element)[8:-2] for element in individual_types])

        message_received_type = str(type(parameter))[8:-2]

        error_message_part_1 = f"Parameter '{parameter_name}': Expected {message_expected_type},"
        error_message_part_2 = f" got {message_received_type} instead"
        full_error_message = error_message_part_1 + error_message_part_2

        raise TypeError(full_error_message)
