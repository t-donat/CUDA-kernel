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
        result_is_correct = np.isclose(calculated_result, expected_result).all()
    except ValueError as value_err:
        print("Matrix dimension missmatch!")
        print(f"Message: {value_err}")
        result_is_correct = False

    if result_is_correct:
        print("Passed!")

    else:
        print("Failed!")
        print(np.where(np.invert(np.isclose(calculated_result, expected_result))))
        print((calculated_result-expected_result)[np.invert(np.isclose(calculated_result, expected_result))])

    successes.append(result_is_correct)
    print("--------------------------------------\n")

if all(successes):
    print("All tests passed successfully!")



