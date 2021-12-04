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

import cunumeric as np


def initialize(N):
    print("Initializing stencil grid...")
    grid = np.zeros((N + 2, N + 2, N+2))
    grid[:,:, 0] = -273.15
    grid[:, 0, :] = -273.15
    grid[0,:, :] = -273.15
    grid[:,:, -1] = 273.15
    grid[:, -1, :] = 273.15
    grid[-1,:, :] = 273.15
 

    return grid


def run(grid, I, N):  # noqa: E741
    print("Running Jacobi 27 stencil...")

    #one
    g000 = grid[0:-2, 0:-2, 0:-2]
    g001 = grid[0:-2, 0:-2, 1:-1]
    g002 = grid[0:-2, 0:-2, 2:  ]

    g010 = grid[0:-2, 1:-1, 0:-2]
    g011 = grid[0:-2, 1:-1, 1:-1]
    g012 = grid[0:-2, 1:-1, 2:  ]

    g020 = grid[0:-2, 2:  , 0:-2]
    g021 = grid[0:-2, 2:  , 1:-1]
    g022 = grid[0:-2, 2:  , 2:  ]

    #two
    g100 = grid[1:-1, 0:-2, 0:-2]
    g101 = grid[1:-1, 0:-2, 1:-1]
    g102 = grid[1:-1, 0:-2, 2:  ]

    g110 = grid[1:-1, 1:-1, 0:-2]
    g111 = grid[1:-1, 1:-1, 1:-1]
    g112 = grid[1:-1, 1:-1, 2:  ]

    g120 = grid[1:-1, 2:  , 0:-2]
    g121 = grid[1:-1, 2:  , 1:-1]
    g122 = grid[1:-1, 2:  , 2:  ]

    #three
    g200 = grid[2:  , 0:-2, 0:-2]
    g201 = grid[2:  , 0:-2, 1:-1]
    g202 = grid[2:  , 0:-2, 2:  ]

    g210 = grid[2:  , 1:-1, 0:-2]
    g211 = grid[2:  , 1:-1, 1:-1]
    g212 = grid[2:  , 1:-1, 2:  ]

    g220 = grid[2:  , 2:  , 0:-2]
    g221 = grid[2:  , 2:  , 1:-1]
    g222 = grid[2:  , 2:  , 2:  ]

    for i in range(I):
        g00 = g000 + g001 + g002 
        g01 = g010 + g011 + g012 
        g02 = g020 + g021 + g022 
        g10 = g100 + g101 + g102 
        g11 = g110 + g111 + g112 
        g12 = g120 + g121 + g122 
        g20 = g200 + g201 + g202 
        g21 = g210 + g211 + g212 
        g22 = g220 + g221 + g222 

        g0 = g00 + g01 + g02
        g1 = g10 + g11 + g12
        g2 = g20 + g21 + g22
         
        res = g0 + g1 + g2
        work = 0.037 * res
        g111[:] = work
    total = np.sum(g111)
    return total / (N ** 2)


def run_stencil(N, I, timing):  # noqa: E741
    start = datetime.datetime.now()
    grid = initialize(N)
    average = run(grid, I, N)
    # This will sync the timing because we will need to wait for the result
    assert not math.isnan(average)
    stop = datetime.datetime.now()
    print("Average energy is %.8g" % average)
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        print("Elapsed Time: " + str(total) + " ms")
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
    args = parser.parse_args()
    run_benchmark(
        run_stencil, args.benchmark, "Stencil", (args.N, args.I, args.timing)
    )
