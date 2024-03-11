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


def check_result(a, q, r):
    print("Checking result...")

    if num.allclose(a, num.matmul(q, r)):
        print("PASS!")
    else:
        print("FAIL!")
        assert False


def qr(m, n, dtype, perform_check, timing):
    a = num.random.rand(m, n).astype(dtype=dtype)

    timer.start()
    q, r = num.linalg.qr(a)
    total = timer.stop()

    if perform_check:
        check_result(a, q, r)

    if timing:
        print(f"Elapsed Time: {total} ms")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-m",
        "--rows",
        type=int,
        default=10,
        dest="m",
        help="number of rows in the matrix",
    )
    parser.add_argument(
        "-n",
        "--cols",
        type=int,
        default=10,
        dest="n",
        help="number of cols in the matrix",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="float64",
        choices=["float32", "float64", "complex64", "complex128"],
        dest="dtype",
        help="data type",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="compare result to numpy",
    )
    args, num, timer = parse_args(parser)

    run_benchmark(
        qr,
        args.benchmark,
        "QR",
        (args.m, args.n, args.dtype, args.check, args.timing),
    )
