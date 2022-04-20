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

"""
cuNumeric
=====

Provides a distributed task-parallel implementation of the Numpy interface
with GPU acceleration.

:meta private:
"""

import numpy as _np

from cunumeric import linalg, random
from cunumeric.array import ndarray
from cunumeric.module import *
from cunumeric._ufunc import *
from cunumeric.window import bartlett, blackman, hamming, hanning, kaiser
from cunumeric.coverage import clone_module

clone_module(_np, globals())

del clone_module
del _np
