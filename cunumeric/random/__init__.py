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

import sys as _sys

import numpy.random as _nprandom
from cunumeric.random.random import *
from cunumeric.utils import (
    add_missing_attributes as _add_missing_attributes,
)

_thismodule = _sys.modules[__name__]

# map any undefined attributes to numpy
_add_missing_attributes(_nprandom, _thismodule)

del _add_missing_attributes
