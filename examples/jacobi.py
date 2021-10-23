#!/usr/bin/env python

# Copyright 2021 NVIDIA Corporation
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

from __future__ import print_function

import argparse
import datetime
import math

from benchmark import run_benchmark

import cunumeric as np


def generate_random(N):
    print("Generating %dx%d system..." % (N, N))
    # Generate a random matrix
    A = np.random.rand(N, N)
    # Make sure that it is diagonally dominate
    A = A + N * np.eye(N)
    # Generate a random vector
    b = np.random.rand(N)
    return A, b


def solve(A, b, iters, verbose):
    print("Solving system...")
    x = np.zeros(A.shape[1])
    d = np.diag(A)
    R = A - np.diag(d)
    for i in range(iters):
        x = (b - np.dot(R, x)) / d
    return x


def check(A, x, b):
    print("Checking result...")
    if np.allclose(A.dot(x), b):
        print("PASS!")
    else:
        print("FAIL!")


def run_jacobi(N, iters, perform_check, timing, verbose):
    start = datetime.datetime.now()
    A, b = generate_random(N)
    x = solve(A, b, iters, verbose)
    if perform_check:
        check(A, x, b)
    else:
        # Need a synchronization here for timing
        assert not math.isnan(np.sum(x))
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--check",
        dest="check",
        action="store_true",
        help="check the result of the solve",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=1000,
        dest="iters",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="print verbose output",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 - "
        "normal execution)",
    )

    args = parser.parse_args()
    run_benchmark(
        run_jacobi,
        args.benchmark,
        "Jacobi",
        (args.N, args.iters, args.check, args.timing, args.verbose),
    )
