# Copyright 2022 NVIDIA Corporation
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
import sys

import numpy as np
import pytest
from utils.random import ModuleGenerator, assert_distribution

import cunumeric as num


def test_permutation_int():
  count = 1024
  p = num.random.permutation(count)
  p.sort()
  assert(num.linalg.norm(p - np.arange(count)) == 0.0)


def test_permutation_array():
  count = 1024
  x = num.arange(count)
  p = num.random.permutation(x)
  assert(num.linalg.norm(x-p) != 0.0)
  p.sort()
  assert(num.linalg.norm(x-p) == 0.0)


def test_shuffle():
  count = 16
  p = num.arange(count)
  x = num.arange(count)
  num.random.shuffle(x)
  assert(num.linalg.norm(x-p) != 0.0)
  x.sort()
  assert(num.linalg.norm(x-p) == 0.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
