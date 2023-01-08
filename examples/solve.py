#!/usr/bin/env python

# Copyright 2022 NVIDIA Corporation
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

from benchmark import parse_args, run_benchmark


def solve(m, n, nrhs, dtype):
    a = np.random.rand(m, n).astype(dtype=dtype)
    b = np.random.rand(n, nrhs).astype(dtype=dtype)

    timer.start()
    np.linalg.solve(a, b)
    total = timer.stop()

    print(f"Elapsed Time: {total} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--num_rows",
        type=int,
        default=10,
        dest="m",
        help="number of rows in the matrix",
    )
    parser.add_argument(
        "-n",
        "--num_cols",
        type=int,
        default=10,
        dest="n",
        help="number of columns in the matrix",
    )
    parser.add_argument(
        "-s",
        "--nrhs",
        type=int,
        default=1,
        dest="nrhs",
        help="number of right hand sides",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="float64",
        choices=["float32", "float64", "complex64", "complex128"],
        dest="dtype",
        help="data type",
    )
    args, np, timer = parse_args(parser)

    run_benchmark(
        solve,
        args.benchmark,
        "Solve",
        (args.m, args.n, args.nrhs, args.dtype),
    )
