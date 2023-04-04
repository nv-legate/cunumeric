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
from enum import IntEnum

from benchmark import CuNumericTimer, parse_args, run_benchmark

NUM_ITERS = 10
WARMUP_ITER = 2

vol_start = 0.1
vol_step = 0.01
t_start = 0.5
t_step = 1.0 / (365 * 10)
money_start = -0.4
money_step = 0.01


RISKFREE = 0.02
S0 = 100.0
N_GREEKS = 7


class Greeks(IntEnum):
    PREM = (0,)
    DELTA = (1,)
    VEGA = (2,)
    GAMMA = (3,)
    VANNA = (4,)
    VOLGA = (5,)
    THETA = 6


def initialize(n_vol_steps, n_t_steps, n_money_steps, D):
    steps = (n_t_steps, n_vol_steps, n_money_steps)

    CALL = np.zeros((N_GREEKS,) + steps, dtype=D)
    PUT = np.zeros((N_GREEKS,) + steps, dtype=D)
    S = np.full(steps, S0, dtype=D)
    temp_arr = np.arange((n_vol_steps * n_t_steps * n_money_steps), dtype=int)
    k_temp = (temp_arr % n_money_steps) * money_step
    k_temp = k_temp.reshape(
        (
            n_t_steps,
            n_vol_steps,
            n_money_steps,
        )
    )
    K = (k_temp + (1 + money_start)) * S0

    t_temp = (temp_arr % (n_vol_steps * n_money_steps)) * vol_step
    t_temp = t_temp.reshape(
        (
            n_t_steps,
            n_vol_steps,
            n_money_steps,
        )
    )
    T = t_temp + t_start
    R = 0.02
    V = np.full((n_t_steps, n_vol_steps, n_money_steps), vol_start, dtype=D)
    for i in range(n_vol_steps):
        V[:, i, :] += i * vol_step

    return CALL, PUT, S, K, T, R, V


# Cumulative distribution function
# https://en.wikipedia.org/wiki/Cumulative_distribution_function
def normCDF(d):
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


# Probability density function.
# https://en.wikipedia.org/wiki/Probability_density_function
def normPDF(d):
    RSQRT2PI = 0.39894228040143267793994605993438
    return RSQRT2PI * np.exp(-0.5 * d * d)


# https://en.wikipedia.org/wiki/Black–Scholes_model
def black_scholes(out, S, K, R, T, V, CP, greek):
    stdev = V * np.sqrt(T)
    df = np.exp(-R * T)
    d1 = (np.log(S / K) + (R + 0.5 * V * V) * T) / stdev
    d2 = d1 - stdev
    nd1 = normCDF(CP * d1)
    nd2 = normCDF(CP * d2)

    if greek == Greeks.PREM:
        out[...] = CP * (S * nd1 - K * df * nd2)
    elif greek == Greeks.DELTA:
        out[...] = CP * nd1
    elif greek == Greeks.VEGA:
        out[...] = S * np.sqrt(T) * normPDF(d1)
    elif greek == Greeks.GAMMA:
        out[...] = normPDF(d1) / (S * V * np.sqrt(T))
    elif greek == Greeks.VANNA:
        out[...] = -d2 * normPDF(d1) / V
    elif greek == Greeks.VOLGA:
        out[...] = S * np.sqrt(T) * d1 * d2 * normPDF(d1) / V
    elif greek == Greeks.THETA:
        out[...] = -(
            0.5 * S * V / np.sqrt(T) * normPDF(d1) + CP * R * df * K * nd2
        )
    else:
        raise RuntimeError("Wrong greek name is passed")


def run_black_scholes(n_vol_steps, n_t_steps, n_money_steps):
    timer = CuNumericTimer()
    print("Start black_scholes")
    CALL, PUT, S, K, T, R, V = initialize(
        n_vol_steps, n_t_steps, n_money_steps, np.float32
    )

    print("After the initialization")
    for i in range(NUM_ITERS):
        if i == WARMUP_ITER:
            timer.start()
        for g in Greeks:
            black_scholes(CALL[g.value], S, K, R, T, V, 1, g)
            black_scholes(PUT[g.value], S, K, R, T, V, -1, g)

    total = (timer.stop()) / (NUM_ITERS - WARMUP_ITER)
    print(f"Elapsed Time: {total} ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--vol_steps",
        type=int,
        default=40,
        dest="n_vol_steps",
        help="number of voltivity steps",
    )

    parser.add_argument(
        "-t",
        "--time_steps",
        type=int,
        default=3650,
        dest="n_time_steps",
        help="number of time steps",
    )
    parser.add_argument(
        "-m",
        "--money_steps",
        type=int,
        default=60,
        dest="n_money_steps",
        help="number of money steps",
    )

    args, np, timer = parse_args(parser)

    run_benchmark(
        run_black_scholes,
        args.benchmark,
        "Black Scholes",
        (args.n_vol_steps, args.n_time_steps, args.n_money_steps),
    )
