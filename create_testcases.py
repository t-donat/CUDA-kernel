import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Create random matrices W and X as well the result Z = W * X")

parser.add_argument("-N", "--N",
                    dest="N",
                    default="64",
                    type=int,
                    help="First dimension of first matrix")

parser.add_argument("-K", "--K",
                    dest="K",
                    default="32",
                    type=int,
                    help="The dimension the two matrices share")

parser.add_argument("-M", "--M",
                    dest="M",
                    default="64",
                    type=int,
                    help="Second dimension of second matrix")

args = parser.parse_args()

N = args.N
K = args.K
M = args.M

# create matrices
W = np.random.randn(N, K).astype(np.float32) * 10
X = np.where(np.random.rand(K, M) > 0.8, 1.0, 0.0)

# calculate results for test_case
expected_result = np.dot(W, X)


os.makedirs("./data/inputs", exist_ok=True)
os.makedirs("./data/outputs", exist_ok=True)

np.savetxt("./data/inputs/W.csv", W, delimiter=",")
np.savetxt("./data/inputs/X.csv", X, delimiter=",")
np.savetxt("./data/outputs/expected_result.csv", expected_result, delimiter=",")

# bool_result = np.dot(W, X_boolean)
# if not np.isclose(float_result, bool_result).all():
#    raise ValueError("The error between the boolean and float result has exceeded the tolerance!")



