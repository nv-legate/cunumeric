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

import random

import numpy

import cunumeric


# Code written using the NumPy interface
def step1(np, n):
    return np.ones(n), np.ones(n)


def step2(np, x, y):
    return np.dot(x, y)


# Malicious adoption strategy
numpy_likes = [numpy, cunumeric]

for x in range(10):
    x, y = step1(random.choice(numpy_likes), 1000000)
    step2(random.choice(numpy_likes), x, y)
