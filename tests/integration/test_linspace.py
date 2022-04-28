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


def test_basic():
    x = np.linspace(2.0, 3.0, num=5)
    y = num.linspace(2.0, 3.0, num=5)
    assert np.array_equal(x, y)


def test_endpoint():
    x = np.linspace(2.0, 3.0, num=5, endpoint=False)
    y = num.linspace(2.0, 3.0, num=5, endpoint=False)
    assert np.array_equal(x, y)


def test_retstep():
    x = np.linspace(2.0, 3.0, num=5, retstep=True)
    y = np.linspace(2.0, 3.0, num=5, retstep=True)
    assert np.array_equal(x[0], y[0])
    assert x[1] == y[1]


def test_axis():
    x = np.array([[0, 1], [2, 3]])
    y = np.array([[4, 5], [6, 7]])
    xp = num.array(x)
    yp = num.array(y)

    z = np.linspace(x, y, num=5, axis=0)
    w = num.linspace(xp, yp, num=5, axis=0)
    assert np.array_equal(z, w)

    z = np.linspace(x, y, num=5, axis=1)
    w = num.linspace(xp, yp, num=5, axis=1)
    assert np.array_equal(z, w)

    z = np.linspace(x, y, num=5, axis=2)
    w = num.linspace(xp, yp, num=5, axis=2)
    assert np.array_equal(z, w)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
