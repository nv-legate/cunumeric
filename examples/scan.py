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
import datetime

import numpy as np
from benchmark import run_benchmark

import cunumeric as num


def initialize(shape, dt, axis):
    if dt == "int":
        A = np.random.randint(1000, size=shape).astype(np.int32)
        if axis is None:
            B = np.zeros(shape=A.size, dtype=np.int32)
        else:
            B = np.zeros(shape=shape, dtype=np.int32)
    elif dt == "float":
        A = np.random.random(shape).astype(np.float32)
        # insert NAN at second element
        if len(shape) == 1:
            A[1] = np.nan
        elif len(shape) == 2:
            A[1, 1] = np.nan
        elif len(shape) == 3:
            A[1, 1, 1] = np.nan
        elif len(shape) == 4:
            A[1, 1, 1, 1] = np.nan

        if axis is None:
            B = np.zeros(shape=A.size, dtype=np.float32)
        else:
            B = np.zeros(shape=shape, dtype=np.float32)
    else:
        A = (
            np.random.random(shape).astype(np.float32)
            + np.random.random(shape).astype(np.float32) * 1j
        )
        if len(shape) == 1:
            A[1] = np.nan
        elif len(shape) == 2:
            A[1, 1] = np.nan
        elif len(shape) == 3:
            A[1, 1, 1] = np.nan
        elif len(shape) == 4:
            A[1, 1, 1, 1] = np.nan

        if axis is None:
            B = np.zeros(shape=A.size, dtype=np.complex64)
        else:
            B = np.zeros(shape=shape, dtype=np.complex64)

    return A, B


def check_scan(OP, A, B, ax):
    C = np.zeros(shape=B.shape, dtype=B.dtype)
    if OP == "cumsum":
        np.cumsum(A, out=C, axis=ax)
    elif OP == "cumprod":
        np.cumprod(A, out=C, axis=ax)
    elif OP == "nancumsum":
        np.nancumsum(A, out=C, axis=ax)
    elif OP == "nancumprod":
        np.nancumprod(A, out=C, axis=ax)

    print("Checking result...")
    if np.allclose(B, C, equal_nan=True):
        print("PASS!")
    else:
        print("FAIL!")
        print("INPUT    : " + str(A))
        print("CUNUMERIC: " + str(B))
        print("NUMPY    : " + str(C))
        assert False


def run_scan(OP, shape, dt, ax, check):
    print("Problem Size:    shape=" + str(shape))

    # axis handling
    if ax is not None:
        assert ax < len(shape) and ax >= 0

    print("Problem Type:    OP=" + OP)
    print("Axis:            axis=" + str(ax))
    print("Data type:       dtype=" + dt + "32")
    A, B = initialize(shape=shape, dt=dt, axis=ax)
    start = datetime.datetime.now()

    # op handling
    if OP == "cumsum":
        num.cumsum(A, out=B, axis=ax)
    elif OP == "cumprod":
        num.cumprod(A, out=B, axis=ax)
    elif OP == "nancumsum":
        num.nancumsum(A, out=B, axis=ax)
    elif OP == "nancumprod":
        num.nancumprod(A, out=B, axis=ax)
    else:
        assert False

    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    print("Elapsed Time:   " + str(total) + "ms")
    # error checking
    if check:
        check_scan(OP, A, B, ax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--shape",
        type=int,
        nargs="+",
        default=[10],
        dest="shape",
        help="array reshape (default '[10]')",
    )
    parser.add_argument(
        "-t",
        "--datatype",
        default="int",
        choices=["int", "float", "complex"],
        dest="dt",
        help="data type (default int)",
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=int,
        default=None,
        dest="axis",
        help="scan axis (default None)",
    )
    parser.add_argument(
        "-o",
        "--operation",
        default="cumsum",
        choices=["cumsum", "cumprod", "nancumsum", "nancumprod"],
        dest="OP",
        help="operation, can be either cumsum (default), cumprod, "
        "nancumsum, nancumprod",
    )
    parser.add_argument(
        "-c",
        "--check",
        dest="check",
        action="store_true",
        help="check the result of the solve",
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
        run_scan,
        args.benchmark,
        "Scan",
        (
            args.OP,
            args.shape,
            args.dt,
            args.axis,
            args.check,
        ),
    )
