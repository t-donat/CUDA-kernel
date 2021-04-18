import numpy as np
from pathlib import Path

output_folder = Path("./data/outputs")
expected_result = np.genfromtxt(output_folder.joinpath("expected_result.csv"), delimiter=",")
successes = []

for file in output_folder.glob("*.csv"):
    if file.name == "expected_result.csv":
        continue
    calculated_result = np.genfromtxt(file, delimiter=",")
    print(f"Test case for {file.name}:")
    print("--------------------------------------")

    try:
        correct_result = np.isclose(calculated_result, expected_result).all()
    except ValueError as value_err:
        print("Matrix dimension missmatch!")
        print(f"Message: {value_err}")
        correct_result = False

    if correct_result:
        print("Passed!")

    else:
        print("Failed!")

    successes.append(correct_result)
    print("--------------------------------------\n")

if all(successes):
    print("All tests passed successfully!")



