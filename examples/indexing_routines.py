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
import gc
import math

from benchmark import run_benchmark

try:
    import cunumeric as np
except ImportError:
    import legate.numpy as np


def compute_diagonal(steps, N, timing):
    A1 = np.ones((N,), dtype=int)
    print("measuring diagonal")
    start = datetime.datetime.now()
    for step in range(steps):
        A2 = np.diag(A1)
        A1 = np.diag(A2)
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        flops = 0
        print("Total Flops:      " + str(flops / 1e9) + " GFLOPS/iter")
        space = (N * N + N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for diagonal: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "bandwidth:               "
            + str(space * 1000.0 / (average))
            + " GB/s"
        )
    return total


def compute_choose(steps, N, timing):
    print("measuring choose")
    A1 = np.ones((N,), dtype=int)
    B = np.arange(N, dtype=int)
    C1 = B % N
    C2 = C1.reshape(
        (
            int(N / 100),
            int(100),
        )
    )
    start = datetime.datetime.now()
    for step in range(steps):
        C1 = np.choose(C1, A1, mode="clip")
        C2 = C1.reshape(
            (
                int(N / 100),
                int(100),
            )
        )
        np.choose(C2, A1, mode="wrap")
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        flops = 0
        print("Total Flops:      " + str(flops / 1e9) + " GFLOPS/iter")
        space = 3 * N * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for choose: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "FLOPS/s:                 "
            + str(flops / (average * 1e6))
            + " GFLOPS/s"
        )
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def compute_repeat(steps, N, timing):
    A2 = np.ones(
        (
            N,
            N,
        ),
        dtype=int,
    )
    print("measuring repeat")
    start = datetime.datetime.now()
    for step in range(steps):
        R = np.repeat(int(1), N)
        A2[:] = np.repeat(A2, R, axis=1)
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        flops = 0
        print("Total Flops:      " + str(flops / 1e9) + " GFLOPS/iter")
        space = (N * N + N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for repeat: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "FLOPS/s:                 "
            + str(flops / (average * 1e6))
            + " GFLOPS/s"
        )
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def compute_advanced_indexing_1d(steps, N, timing):
    A1 = np.ones((N,), dtype=int)
    B = np.arange(N, dtype=int)
    print("measuring advanced_indexing 1D")
    indx = B % 10
    indx_bool = (B % 2).astype(bool)
    start = datetime.datetime.now()
    for step in range(steps):
        A1[indx] = 10  # 1 copy
        A1[indx_bool] = 12  # 1 AI and 1 copy
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        flops = 0
        print("Total Flops:      " + str(flops / 1e9) + " GFLOPS/iter")
        space = (3 * N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for advanced_indexing: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "FLOPS/s:                 "
            + str(flops / (average * 1e6))
            + " GFLOPS/s"
        )
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def compute_advanced_indexing_2d(steps, N, timing):
    A2 = np.ones((N, N), dtype=int)
    B = np.arange(N, dtype=int)
    print("measuring advanced_indexing 2D")
    indx = B % 10
    indx_bool = (B % 2).astype(bool)
    indx2d_bool = (A2 % 2).astype(bool)
    start = datetime.datetime.now()
    for step in range(steps):
        A2[indx_bool, indx_bool] = 11  # one ZIP and 1 copy = N+N*N
        A2[:, indx] = 12  # one ZIP and 3 copies = N+3*N*N
        A2[indx2d_bool] = 13  # 1 copy and one AI task = 2* N*N
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        flops = 0
        print("Total Flops:      " + str(flops / 1e9) + " GFLOPS/iter")
        space = (6 * N * N + 2 * N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for advanced_indexing: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "FLOPS/s:                 "
            + str(flops / (average * 1e6))
            + " GFLOPS/s"
        )
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def compute_advanced_indexing_3d(steps, N, timing):
    A3 = np.ones(
        (
            N,
            int(N / 100),
            100,
        ),
        dtype=int,
    )
    B = np.arange(N, dtype=int)
    print("measuring advanced_indexing_3d")
    indx = B % 10
    indx3d_bool = (A3 % 2).astype(bool)
    start = datetime.datetime.now()
    for step in range(steps):
        A3[indx, :, indx] = 15  # 1 ZIP and 3 copy = N+3N*N
        A3[indx3d_bool] = 16  # 1 copy and 1 AI task = 2*N*N
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        flops = 0
        print("Total Flops:      " + str(flops / 1e9) + " GFLOPS/iter")
        space = (5 * N * N + N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for advanced_indexing: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "FLOPS/s:                 "
            + str(flops / (average * 1e6))
            + " GFLOPS/s"
        )
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def run_indexing_routines(
    N,
    steps,
    timing,
    verbose,
    routine,
):
    # simple operation to warm up the library
    assert not math.isnan(np.sum(np.zeros((N, N)).dot(np.zeros((N,)))))
    gc.collect()
    time = 0
    if routine == "diagonal" or routine == "all":
        time += compute_diagonal(steps, N, timing)
    if routine == "choose" or routine == "all":
        time += compute_choose(steps, N, timing)
    if routine == "repeat" or routine == "all":
        time += compute_repeat(steps, N, timing)
    if routine == "ai1" or routine == "all":
        time += compute_advanced_indexing_1d(steps, N, timing)
    if routine == "ai2" or routine == "all":
        time += compute_advanced_indexing_2d(steps, N, timing)
    if routine == "ai3" or routine == "all":
        time += compute_advanced_indexing_3d(steps, N, timing)
    if timing:
        print("Total Elapsed Time: " + str(time) + " ms")
    return time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100,
        dest="I",
        help="number of iterations to run the algorithm for",
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
    parser.add_argument(
        "-r",
        "--routine",
        default="all",
        dest="routine",
        help="name of the index routine to test",
    )
    args, unknown = parser.parse_known_args()
    print("Warning, unrecognized arguments: ", unknown)
    run_benchmark(
        run_indexing_routines,
        args.benchmark,
        "Core",
        (args.N, args.I, args.timing, args.verbose, args.routine),
    )
