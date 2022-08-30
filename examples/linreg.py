#!/usr/bin/env python

# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

from benchmark import parse_args, run_benchmark, time


def initialize(N, F, T):
    # We'll generate some random inputs here
    # since we don't need it to converge
    x = np.random.randn(N, F).astype(T)
    y = np.random.random(N).astype(T)
    return x, y


def run_linear_regression(N, F, T, I, warmup, S, B):  # noqa: E741
    print("Running linear regression...")
    print("Number of data points: " + str(N) + "K")
    print("Number of features: " + str(F))
    print("Number of iterations: " + str(I))

    learning_rate = 1e-5
    features, target = initialize(N * 1000, F, T)
    if B:
        intercept = np.ones((features.shape[0], 1), dtype=T)
        features = np.hstack((intercept, features))
    weights = np.zeros(features.shape[1], dtype=T)

    start = time()
    for step in range(-warmup, I):
        if step == 0:
            start = time()
        scores = np.dot(features, weights)
        error = scores - target
        gradient = -(1.0 / len(features)) * error.dot(features)
        weights += learning_rate * gradient
        if step >= 0 and step % S == 0:
            print(
                "Error of step "
                + str(step)
                + ": "
                + str(np.sum(np.power(error, 2)))
            )
    stop = time()

    total = (stop - start) / 1000.0
    print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-B",
        "--intercept",
        dest="B",
        action="store_true",
        help="include an intercept in the calculation",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=int,
        default=32,
        dest="F",
        help="number of features for each input data point",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1000,
        dest="I",
        help="number of iterations to run the algorithm for",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=5,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="N",
        help="number of elements in the data set in thousands",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=32,
        dest="P",
        help="precision of the computation in bits",
    )
    parser.add_argument(
        "-s",
        "--sample",
        type=int,
        default=100,
        dest="S",
        help="number of iterations between sampling the log likelihood",
    )

    args, np = parse_args(parser)

    if args.P == 16:
        run_benchmark(
            run_linear_regression,
            args.benchmark,
            "LINREG(H)",
            (args.N, args.F, np.float16, args.I, args.warmup, args.S, args.B),
        )
    elif args.P == 32:
        run_benchmark(
            run_linear_regression,
            args.benchmark,
            "LINREG(S)",
            (args.N, args.F, np.float32, args.I, args.warmup, args.S, args.B),
        )
    elif args.P == 64:
        run_benchmark(
            run_linear_regression,
            args.benchmark,
            "LINREG(D)",
            (args.N, args.F, np.float64, args.I, args.warmup, args.S, args.B),
        )
    else:
        raise TypeError("Precision must be one of 16, 32, or 64")
