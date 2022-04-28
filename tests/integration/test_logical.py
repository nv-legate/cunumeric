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


def test_any_basic():
    assert num.array_equal(num.any([-1, 4, 5]), np.any([-1, 4, 5]))

    x = [5, 10, 0, 100]
    cx = num.array(x)
    assert num.array_equal(num.any(cx), np.any(x))

    y = [[0, 0], [0, 0]]
    cy = num.array(y)
    assert num.array_equal(num.any(cy), np.any(y))


def test_any_axis():
    x = np.array([[True, True, False], [True, True, True]])
    cx = num.array(x)

    assert num.array_equal(num.any(cx), np.any(x))
    assert num.array_equal(num.any(cx, axis=0), np.any(x, axis=0))


def test_all_basic():
    assert num.array_equal(num.all([-1, 4, 5]), np.all([-1, 4, 5]))

    x = [5, 10, 0, 100]
    cx = num.array(x)
    assert num.array_equal(num.all(cx), np.all(x))

    y = [[0, 0], [0, 0]]
    cy = num.array(y)
    assert num.array_equal(num.all(cy), np.all(y))


def test_all_axis():
    x = np.array([[True, True, False], [True, True, True]])
    cx = num.array(x)

    assert num.array_equal(num.all(cx), np.all(x))
    assert num.array_equal(num.all(cx, axis=0), np.all(x, axis=0))


def test_nan():
    assert num.equal(num.all(num.nan), np.all(np.nan))
    assert num.equal(num.any(num.nan), np.any(np.nan))

    assert num.array_equal(num.all(num.nan), np.all(np.nan))
    assert num.array_equal(num.any(num.nan), np.any(np.nan))


@pytest.mark.skip
def test_where():
    x = np.array([[True, True, False], [True, True, True]])
    y = np.array([[True, False], [True, True]])
    cy = num.array(y)

    assert num.array_equal(
        num.all(cy, where=[True, False]), np.all(x, where=[True, False])
    )
    assert num.array_equal(
        num.any(cy, where=[[True], [False]]),
        np.any(x, where=[[True], [False]]),
    )


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
