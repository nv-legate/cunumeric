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
import math

from benchmark import run_benchmark

try:
    from legate.timing import time
except ImportError:
    from time import perf_counter_ns

    def time():
        return perf_counter_ns() / 1000.0


def initialize(N):
    print("Initializing stencil grid...")
    grid = np.zeros((N + 2, N + 2))
    grid[:, 0] = -273.15
    grid[:, -1] = -273.15
    grid[-1, :] = -273.15
    grid[0, :] = 40.0
    return grid


def run(grid, I, N):  # noqa: E741
    print("Running Jacobi stencil...")
    center = grid[1:-1, 1:-1]
    north = grid[0:-2, 1:-1]
    east = grid[1:-1, 2:]
    west = grid[1:-1, 0:-2]
    south = grid[2:, 1:-1]
    for i in range(I):
        average = center + north + east + west + south
        work = 0.2 * average
        # delta = np.sum(np.absolute(work - center))
        center[:] = work
    total = np.sum(center)
    return total / (N**2)


def run_stencil(N, I, timing):  # noqa: E741
    grid = initialize(N)
    start = time()
    average = run(grid, I, N)
    stop = time()
    print("Average energy is %.8g" % average)
    total = (stop - start) / 1000.0
    assert not math.isnan(average)
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
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 "
        "- normal execution)",
    )
    parser.add_argument(
        "--package",
        dest="package",
        choices=["legate", "numpy", "cupy"],
        type=str,
        default="legate",
        help="NumPy package to use (legate, numpy, or cupy)",
    )
    parser.add_argument(
        "--cupy-allocator",
        dest="cupy_allocator",
        choices=["default", "off", "managed"],
        type=str,
        default="default",
        help="cupy allocator to use (default, off, or managed)",
    )

    args = parser.parse_args()

    if args.package == "legate":
        import cunumeric as np
    elif args.package == "cupy":
        import cupy as np

        if args.cupy_allocator == "off":
            np.cuda.set_allocator(None)
            print("Turning off memory pool")
        elif args.cupy_allocator == "managed":
            np.cuda.set_allocator(
                np.cuda.MemoryPool(np.cuda.malloc_managed).malloc
            )
            print("Using managed memory pool")
    elif args.package == "numpy":
        import numpy as np

    run_benchmark(
        run_stencil, args.benchmark, "Stencil", (args.N, args.I, args.timing)
    )
