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


def initialize(N):
    print("Initializing stencil grid...")
    grid = np.zeros((N + 2, N + 2))
    grid[:, 0] = -273.15
    grid[:, -1] = -273.15
    grid[-1, :] = -273.15
    grid[0, :] = 40.0
    return grid


def run_stencil(N, I, warmup, timing):  # noqa: E741
    grid = initialize(N)

    print("Running Jacobi stencil...")
    center = grid[1:-1, 1:-1]
    north = grid[0:-2, 1:-1]
    east = grid[1:-1, 2:]
    west = grid[1:-1, 0:-2]
    south = grid[2:, 1:-1]

    start = time()
    for i in range(I + warmup):
        if i == warmup:
            start = time()
        average = center + north + east + west + south
        work = 0.2 * average
        center[:] = work
    stop = time()

    total = (stop - start) / 1000.0
    if timing:
        print(f"Elapsed Time: {total} ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
        dest="I",
        help="number of iterations to run",
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

    args, np = parse_args()

    run_benchmark(
        run_stencil, "Stencil", (args.N, args.I, args.warmup, args.timing)
    )
