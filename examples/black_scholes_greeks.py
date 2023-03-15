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

from benchmark import parse_args, run_benchmark, CuNumericTimer

import math
import cunumeric as np


#big size
#n_vol_steps = 40
vol_start = 0.1
vol_step = 0.01
#n_t_steps = 365*10
t_start = 0.5
t_step = 1.0/(365*10)
#n_money_steps = 60
money_start = -0.4
money_step = 0.01


#small size
#n_vol_steps = 10
#vol_start = 0.1
#vol_step = 0.01
#n_t_steps = 6
#t_start = 0.5
#t_step = 0.5
#n_money_steps = 1
#money_start = 0
#money_step = 0.1

RISKFREE = 0.02
S0 = 100.0
N_GREEKS=7


def initialize(n_vol_steps, n_t_steps, n_money_steps, D):
    CALL = np.zeros((N_GREEKS, n_t_steps, n_vol_steps, n_money_steps,), dtype = D)
    PUT = np.zeros((N_GREEKS, n_t_steps, n_vol_steps, n_money_steps,), dtype = D)
    S=np.full((n_t_steps, n_vol_steps, n_money_steps,),S0, dtype = D)
    K=np.full((n_t_steps, n_vol_steps, n_money_steps,), (1 + money_start), dtype = D)  
    temp_arr = np.arange((n_vol_steps*n_t_steps*n_money_steps), dtype=int)
    k_temp=(temp_arr%n_money_steps)*money_step
    k_temp = k_temp.reshape((n_t_steps, n_vol_steps, n_money_steps,))
    K+=k_temp
    K=K*S0

    T=np.full((n_t_steps, n_vol_steps, n_money_steps,),t_start, dtype = D)
    t_temp = (temp_arr%(n_vol_steps*n_money_steps))*vol_step
    t_temp = t_temp.reshape((n_t_steps, n_vol_steps, n_money_steps,))
    T+=t_temp
    R=  0.02
    V=np.full((n_t_steps, n_vol_steps, n_money_steps), vol_start, dtype = D)
    for i in range(n_vol_steps):
        V[:,i,:]+=i*vol_step        

    return CALL, PUT, S, K, T, R, V

def normCDF(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    K = 1.0 / (1.0 + 0.2316419 * np.absolute(d))

    cnd = RSQRT2PI * np.exp(- 0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    return np.where(d > 0, 1.0 - cnd, cnd)

def normPDF(d):
    RSQRT2PI = 0.39894228040143267793994605993438;
    return RSQRT2PI * np.exp(- 0.5 * d * d);

def black_scholes ( out , S, K, R, T, V, d1, d2, nd1, nd2, CP, greek):
    EPS = 0.00000001
    stdev = V * np.sqrt(T)
    df = np.exp(-R*T)
    d1 = (np.log(S/K)+(R+0.5*V*V)*T)/stdev
    d2= d1-stdev
    nd1 = normCDF(CP*d1)
    nd2 = normCDF(CP*d2)

    if greek == "PREM":
        out[...] = CP*(S*nd1 - K*df*nd2);
    elif greek == "DELTA":
        out[...] = CP*nd1
    elif greek =="VEGA":
        out[...] = S*np.sqrt(T)*normPDF(d1)
    elif greek == "GAMMA":
        out[...] = normPDF(d1)/(S*V*np.sqrt(T))
    elif greek == "VANNA":
        out[...] = -d2*normPDF(d1)/V
    elif greek == "VOLGA":
        out[...] = S*np.sqrt(T)*d1*d2*normPDF(d1)/V;
    elif greek == "THETA":
        out[...] = -(0.5*S*V/np.sqrt(T)*normPDF(d1)+CP*R*df*K*nd2)
    else:
        RuntimeError("Wrong greek name is passed")


   
greeks = ["PREM", "DELTA", "VEGA", "GAMMA", "VANNA", "VOLGA", "THETA",]

def run_black_scholes(n_vol_steps, n_t_steps, n_money_steps):
    timer = CuNumericTimer()
    print("Start black_scholes")
    CALL, PUT, S, K, T, R, V = initialize(n_vol_steps, n_t_steps, n_money_steps, np.float32)

    d1 = np.zeros_like(S)
    d2= np.zeros_like(S)
    nd1 = np.zeros_like(S)
    nd2= np.zeros_like(S)

    print("After the initialization")
    timer.start()
    for count,g in enumerate(greeks):
        black_scholes(CALL[count],S, K, R, T, V, d1, d2, nd1, nd2,1, g)
        black_scholes(PUT[count],S, K, R, T, V, d1, d2, nd1, nd2, -1, g)

    total = timer.stop()
    print("Elapsed Time: " + str(total) + " ms")
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--vol_tesps",
        type=int,
        default=40,
        dest="n_vol_steps",
        help="number of voltivity steps",
    )

    parser.add_argument(
        "-t",
        "--time_tesps",
        type=int,
        default=3650,
        dest="n_time_steps",
        help="number of time steps",
    )
    parser.add_argument(
        "-m",
        "--money_tesps",
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


