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
from itertools import product

import numpy as np
import pytest

import cunumeric as num

a = num.random.random((10, 10, 10))


# product minus the "diagonal"
def ul_prod(iterable):
    for a, b in product(iterable, repeat=2):
        if a == b:
            continue
        yield (a, b)


def test_basic():
    anp = a.__array__()
    b = num.flip(a)
    bnp = np.flip(anp)

    assert num.array_equal(b, bnp)


AXES = [0, 1, 2]


@pytest.mark.parametrize("axis", AXES)
def test_axis_1d(axis):
    anp = a.__array__()
    b = num.flip(a, axis=axis)
    bnp = np.flip(anp, axis=axis)

    assert num.array_equal(b, bnp)


@pytest.mark.parametrize("axis", ul_prod(AXES), ids=str)
def test_axis_2d(axis):
    anp = a.__array__()
    b = num.flip(a, axis=axis)
    bnp = np.flip(anp, axis=axis)

    assert num.array_equal(b, bnp)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
