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

import math
import sys

if sys.version_info > (3, 0):
    from functools import reduce


# A helper method for benchmarking applications
def run_benchmark(f, samples, name, args):
    if samples > 1:
        results = [f(*args) for s in range(samples)]
        # Remove the largest and the smallest ones
        if samples >= 3:
            results.remove(max(results))
        if samples >= 2:
            results.remove(min(results))
        mean = sum(results) / len(results)
        variance = sum(map(lambda x: (x - mean) ** 2, results)) / len(results)
        stddev = math.sqrt(variance)
        print("-----------------------------------------------")
        print("BENCHMARK RESULTS: " + name)
        print("Total Samples: " + str(samples))
        print("Average Time: " + str(mean) + " ms")
        print("Variance: " + str(variance) + " ms")
        print("Stddev: " + str(stddev) + " ms")
        print(
            "All Results: "
            + reduce(lambda x, y: x + y, map(lambda x: str(x) + ", ", results))
        )
        print("-----------------------------------------------")
    else:
        # Just run the application like normal
        f(*args)
