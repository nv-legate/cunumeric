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

#import pdb
#pdb.set_trace()
import legate.numpy as np
import numpy as npo
#import numpy as np
N=6000
b1 = np.zeros(N)
b2 = np.zeros(N)
b3 = np.zeros(N)
b1.fill(2)
b2.fill(3)
b3.fill(4)

#dummy ops
# dumb hack so the loop starts at nOps=10
# so the window/pipe is empty
b3.fill(4)
b3.fill(4)
b3.fill(4)
b3.fill(4)


#for i in range(10000):
for i in range(10000):
    b3=b1+b2

""
