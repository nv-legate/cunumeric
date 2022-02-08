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

import argparse
import datetime

import numpy
from benchmark import run_benchmark

import cunumeric


def check_sorted(a, a_numpy):
    a_sorted = numpy.sort(a_numpy)
    print("Checking result...")
    if cunumeric.allclose(a_sorted, a):
        print("PASS!")
    else:
        print("FAIL!")
        print("NUMPY    : " + str(a_sorted))
        print("CUNUMERIC: " + str(a))


def run_sort(N, perform_check, timing):

    numpy.random.seed(42)
    a_numpy = numpy.array(
        numpy.random.randint(1000, size=N), dtype=numpy.int32
    )
    a = cunumeric.array(a_numpy)

    start = datetime.datetime.now()
    a_sorted = cunumeric.sort(a)
    stop = datetime.datetime.now()

    if perform_check:
        check_sorted(a_sorted, a_numpy)
    else:
        # do we need to synchronize?
        assert True
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
        "-n",
        "--num",
        type=int,
        default=1000000,
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
        help="number of times to benchmark this application (default 1 - "
        "normal execution)",
    )

    args = parser.parse_args()
    run_benchmark(
        run_sort,
        args.benchmark,
        "Sort",
        (args.N, args.check, args.timing),
    )
