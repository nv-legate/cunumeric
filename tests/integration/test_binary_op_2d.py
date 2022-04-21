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

import numpy as np
import pytest

import cunumeric as num

numpyX = np.array([[1, 2, 3], [4, 5, 6]])
numpyY = np.array([[7, 8, 9], [10, 11, 12]])

x = pytest.fixture(lambda: num.array(numpyX))
y = pytest.fixture(lambda: num.array(numpyY))


def test_add(x, y):
    z = x + y
    assert np.array_equal(z, numpyX + numpyY)

    z = x + 2
    assert np.array_equal(z, numpyX + 2)

    z = 2 + x
    assert np.array_equal(z, 2 + numpyX)


def test_sub(x, y):
    z = x - y
    assert np.array_equal(z, numpyX - numpyY)

    z = x - 2
    assert np.array_equal(z, numpyX - 2)

    z = 2 - x
    assert np.array_equal(z, 2 - numpyX)


def test_div(x, y):
    z = x / y
    assert np.array_equal(z, numpyX / numpyY)

    z = x / 2
    assert np.array_equal(z, numpyX / 2)

    z = 2 / x
    assert np.array_equal(z, 2 / numpyX)


def test_mul(x, y):
    z = x * y
    assert np.array_equal(z, numpyX * numpyY)

    z = x * 2
    assert np.array_equal(z, numpyX * 2)

    z = 2 * x
    assert np.array_equal(z, 2 * numpyX)

    z = x**5
    assert np.array_equal(z, numpyX**5)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
