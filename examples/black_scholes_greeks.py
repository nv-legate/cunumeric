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
EPS = 0.00000001


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



def normPDF(d):
    RSQRT2PI = 0.39894228040143267793994605993438;
    return RSQRT2PI * np.exp(- 0.5 * d * d);

def black_scholes_vec_kernel(d1, d2, nd1, nd2, S, K, T, V, stdev, R,CP, EPS):
    if (math.fabs(V)>EPS) and (math.fabs(T)>EPS) and (math.fabs(K)>EPS) and (math.fabs(S)>EPS):
       d1 = (math.log(S/K)+(R+0.5*V*V)*T)/stdev
       d2=d1-stdev
       cpd1 = CP*d1
       cpd2 = CP*d2
       #manual inlining ndtr
       NPY_SQRT1_2 = 0.707106781186547524400844362104849039
       x = cpd1 * NPY_SQRT1_2
       z = math.fabs(x)

       if z < NPY_SQRT1_2:
           y = 0.5 + 0.5 * math.erf(x)
       else:
           y = 0.5 * math.erfc(z)

           if x > 0:
               y = 1.0 - y
       nd1=y

       #manual inlining ndtr
       x = cpd2 * NPY_SQRT1_2
       z = math.fabs(x)

       if z < NPY_SQRT1_2:
           y = 0.5 + 0.5 * math.erf(x)
       else:
           y = 0.5 * math.erfc(z)

           if x > 0:
               y = 1.0 - y
       nd2=y
    else:
        if (math.fabs(V)<=EPS) or (math.fabs(T)<=EPS) or (math.fabs(K)<=EPS):
            d1 = math.inf
            d2 = math.inf
            nd1 = 1.
            nd2 = 1.
        else:
            d1 = -math.inf
            d2 = -math.inf
            nd1 = 1.
            nd2 = 1.


bs_vec = np.vectorize(black_scholes_vec_kernel,otypes=(float,float,float,float), cache=True)

def black_scholes ( out , S, K, R, T, V, d1, d2, nd1, nd2, df,ind_v, ind_t, CP, greek):

    if greek == "PREM":
        out[...] = CP*(S*nd1 - K*df*nd2);
    elif greek == "DELTA":
        out[...] = CP*nd1
    elif greek =="VEGA":
        out[...] = S*np.sqrt(T)*normPDF(d1)
    elif greek == "GAMMA":
        out[...] = normPDF(d1)/(S*V*np.sqrt(T))
        out[ind_v] =0.
    elif greek == "VANNA":
        out[...] = -d2*normPDF(d1)/V
        out[ind_v] =0.
    elif greek == "VOLGA":
        out[...] = S*np.sqrt(T)*d1*d2*normPDF(d1)/V;
        out[ind_v] =0.
    elif greek == "THETA":
        out[...] = -(0.5*S*V/np.sqrt(T)*normPDF(d1)+CP*R*df*K*nd2)
    else:
        RuntimeError("Wrong greek name is passed")

    if (greek != "PREM"):
        out[ind_t] = 0.

   
greeks = ["PREM", "DELTA", "VEGA", "GAMMA", "VANNA", "VOLGA", "THETA",]
#greeks = ["PREM",]

def run_black_scholes(n_vol_steps, n_t_steps, n_money_steps):
    timer = CuNumericTimer()
    print("Start black_scholes")
    CALL, PUT, S, K, T, R, V = initialize(n_vol_steps, n_t_steps, n_money_steps, np.float32)
    #pre-compute some data for black_scholes
    stdev = V * np.sqrt(T)
    df = np.exp(-R*T)
    ind_v = np.nonzero(np.absolute(V)<EPS)
    ind_t = np.nonzero(np.absolute(T)<EPS)

    d1_call = np.zeros_like(S)
    d2_call= np.zeros_like(S)
    nd1_call = np.zeros_like(S)
    nd2_call= np.zeros_like(S)

    bs_vec(d1_call, d2_call, nd1_call, nd2_call, S, K, T, V,stdev, R, 1, EPS);
    d1_put = np.zeros_like(S)
    d2_put= np.zeros_like(S)
    nd1_put = np.zeros_like(S)
    nd2_put= np.zeros_like(S)

    bs_vec(d1_put, d2_put, nd1_put, nd2_put, S, K, T, V,stdev, R, -1, EPS);

    print("After the initialization")
    timer.start()
    for count,g in enumerate(greeks):
        black_scholes(CALL[count],S, K, R, T, V, d1_call, d2_call, nd1_call, nd2_call, df, ind_v, ind_t, 1, g)
        black_scholes(PUT[count],S, K, R, T, V, d1_put, d2_put, nd1_put, nd2_put, df, ind_v, ind_t, -1, g)

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


