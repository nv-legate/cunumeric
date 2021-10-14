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

import legate.numpy as np


def generate_random(N, min, max, D):
    diff = D(max) - D(min)
    rands = np.random.random(N).astype(D)
    rands = rands * diff
    rands = rands + D(min)
    return rands


def initialize(N, D):
    S = generate_random(N, 5, 30, D)
    X = generate_random(N, 1, 100, D)
    T = generate_random(N, 0.25, 10, D)
    R = 0.02
    V = 0.3
    return S, X, T, R, V


def cnd(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    K = 1.0 / (1.0 + 0.2316419 * np.absolute(d))

    cnd = (
        RSQRT2PI
        * np.exp(-0.5 * d * d)
        * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    )

    return np.where(d > 0, 1.0 - cnd, cnd)


def black_scholes(S, X, T, R, V):
    sqrt_t = np.sqrt(T)
    #d1 = np.log(S / X) + (R + 0.5 * V * V) * T / (V * sqrt_t)
    d1a = np.log(S / X)
    d1b = (R + 0.5 * V * V) 
    d1c = d1b*T
    d1d = d1c/(V * sqrt_t)
    d1 = d1a+d1d

    d2 = d1 - V * sqrt_t
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)
    exp_rt = np.exp(-R * T)
    call_result = S * cnd_d1 - X * exp_rt * cnd_d2
    put_result = X * exp_rt * (1.0 - cnd_d2) - S * (1.0 - cnd_d1)
    return call_result, put_result


def run_black_scholes(N, D):
    print("Running black scholes on %dK options..." % N)
    N *= 1000
    start = datetime.datetime.now()
    S, X, T, R, V = initialize(N, D)
    call, put = black_scholes(S, X, T, R, V)
    # Check the result for NaNs to synchronize before stopping timing
    call_sum = np.sum(call)
    put_sum = np.sum(put)
    assert not math.isnan(call_sum) and not math.isnan(put_sum)
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="N",
        help="number of options to price in thousands",
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
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 - "
        "normal execution)",
    )
    args = parser.parse_args()
    if args.P == 16:
        run_benchmark(
            run_black_scholes,
            args.benchmark,
            "Black Scholes",
            (args.N, np.float16),
        )
    elif args.P == 32:
        run_benchmark(
            run_black_scholes,
            args.benchmark,
            "Black Scholes",
            (args.N, np.float32),
        )
    elif args.P == 64:
        run_benchmark(
            run_black_scholes,
            args.benchmark,
            "Black Scholes",
            (args.N, np.float64),
        )
    else:
        raise TypeError("Precision must be one of 16, 32, or 64")
