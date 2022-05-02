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


def assert_equal(numarr, nparr):
    for resultnp, resultnum in zip(nparr, numarr):
        assert np.array_equal(resultnp, resultnum)


def test_empty_1d():
    assert num.count_nonzero(num.array([])) == 0
    assert num.count_nonzero(num.array([], dtype="?")) == 0
    assert_equal(num.nonzero([]), np.nonzero([]))


def test_empty_2d():
    assert num.count_nonzero(num.array([[], []])) == 0
    assert num.count_nonzero(num.array([[], []], dtype="?")) == 0
    assert_equal(num.nonzero([[], []]), np.nonzero([[], []]))


def test_empty_basic():
    assert num.count_nonzero(num.array([0])) == 0
    assert num.count_nonzero(num.array([0], dtype="?")) == 0
    assert_equal(num.nonzero(num.array([0])), ([],))

    assert num.count_nonzero(num.array([1])) == 1
    assert num.count_nonzero(num.array([1], dtype="?")) == 1
    assert_equal(num.nonzero([1]), np.nonzero([1]))


def test_1d():
    x = num.array([1, 0, 2, -1, 0, 0, 8])
    x_np = np.array([1, 0, 2, -1, 0, 0, 8])
    assert num.count_nonzero(x) == 4
    assert_equal(num.nonzero(x), np.nonzero(x_np))


def test_2d():
    x_lg = num.eye(3)
    x_np = np.eye(3)
    assert num.count_nonzero(x_lg) == np.count_nonzero(x_np)
    assert_equal(num.nonzero(x_lg), np.nonzero(x_np))

    x = num.array([[0, 1, 0] * 5, [2, 0, 3] * 5])
    x_np = np.array([[0, 1, 0] * 5, [2, 0, 3] * 5])
    assert num.count_nonzero(x) == 15
    assert_equal(num.nonzero(x), np.nonzero(x_np))


def test_indexed():
    x_np = np.random.randn(100)
    indices = np.random.choice(
        np.arange(x_np.size), replace=False, size=int(x_np.size * 0.2)
    )
    x_np[indices] = 0
    x = num.array(x_np)
    assert num.count_nonzero(x) == np.count_nonzero(x_np)
    lg_nonzero = num.nonzero(x)
    np_nonzero = np.nonzero(x_np)
    assert_equal(lg_nonzero, np_nonzero)


def test_axis():
    x = num.array(
        [
            [[0, 1], [1, 1], [7, 0], [1, 0], [0, 1]],
            [[3, 0], [0, 3], [0, 0], [2, 2], [0, 19]],
        ]
    )
    x_np = np.array(
        [
            [[0, 1], [1, 1], [7, 0], [1, 0], [0, 1]],
            [[3, 0], [0, 3], [0, 0], [2, 2], [0, 19]],
        ]
    )

    assert_equal(num.nonzero(x), np.nonzero(x_np))
    assert num.count_nonzero(x) == np.count_nonzero(x_np)
    assert num.count_nonzero(x, axis=(0, 1, 2)) == np.count_nonzero(
        x_np, axis=(0, 1, 2)
    )

    # TODO: Put this back once we have per-axis count_nonzero
    # for axis in range(3):
    #    assert_equal(
    #        num.count_nonzero(x, axis=axis),
    #        np.count_nonzero(x_np, axis=axis),
    #    )


def test_deprecated_0d():
    with pytest.deprecated_call():
        assert num.count_nonzero(num.array(0)) == 0
        assert num.count_nonzero(num.array(0, dtype="?")) == 0
        assert_equal(num.nonzero(0), np.nonzero(0))

    with pytest.deprecated_call():
        assert num.count_nonzero(num.array(1)) == 1
        assert num.count_nonzero(num.array(1, dtype="?")) == 1
        assert_equal(num.nonzero(1), np.nonzero(1))

    with pytest.deprecated_call():
        assert_equal(num.nonzero(0), ([],))

    with pytest.deprecated_call():
        assert_equal(num.nonzero(1), ([0],))

    x_np = np.array([True, True])
    x = num.array(x_np)
    assert np.array_equal(x_np.nonzero(), x.nonzero())


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
