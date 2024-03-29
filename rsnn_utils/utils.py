from typing import Any, Union


def check_type(parameter: Any, expected_type: Union[type, tuple[type]], parameter_name: str):

    if hasattr(expected_type, "__origin__") and expected_type.__origin__ == Union:
        # Union, Optional from typing package
        expected_type = expected_type.__args__

    if not isinstance(parameter, expected_type):

        if isinstance(expected_type, type):
            message_expected_type = str(expected_type)[8:-2]

        else:  # if isinstance(expected_type, tuple):
            message_expected_type = ", ".join([str(element)[8:-2] for element in expected_type])

        message_received_type = str(type(parameter))[8:-2]

        error_message_part_1 = f"Parameter '{parameter_name}': Expected {message_expected_type},"
        error_message_part_2 = f" got {message_received_type} instead"
        full_error_message = error_message_part_1 + error_message_part_2

        raise TypeError(full_error_message)


def convert_seconds_to_dhms(duration_in_seconds: Union[int, float]) -> str:
    """Convert a number of seconds into a string of days, hours, minutes and seconds

    Args:
        duration_in_seconds (Union[int, float]): The duration (in seconds) that should be converted into the string
            format

    Returns:
        A string containing the equivalent days, hours, minutes and seconds

    """

    # remove decimal points
    duration_in_seconds = int(duration_in_seconds)

    seconds = duration_in_seconds % 60
    duration_in_minutes = duration_in_seconds // 60

    minutes = duration_in_minutes % 60
    duration_in_hours = duration_in_minutes // 60

    hours = duration_in_hours % 24
    days = duration_in_hours // 24

    return f"{days}-{hours}:{minutes}:{seconds}"
