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

DIM = 5
SIZES = [
    (0,),
    (1),
    (DIM),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]


@pytest.mark.xfail
def test_none_array_compare():
    res_num = num.squeeze(None)  # AttributeError: 'NoneType'
    res_np = np.squeeze(None)  # return None
    assert np.array_equal(res_num, res_np, equal_nan=True)


def test_none_array():
    # numpy returned None
    msg = r"NoneType"
    with pytest.raises(AttributeError, match=msg):
        num.squeeze(None)


def test_num_invalid_axis():
    size = (1, 2, 1)
    a = num.random.randint(low=-10, high=10, size=size)
    msg = r"one"
    with pytest.raises(ValueError, match=msg):
        num.squeeze(a, axis=1)


def test_array_invalid_axis():
    size = (1, 2, 1)
    a = num.random.randint(low=-10, high=10, size=size)
    msg = r"one"
    with pytest.raises(ValueError, match=msg):
        a.squeeze(axis=1)


def test_num_axis_out_bound():
    size = (1, 2, 1)
    a = num.random.randint(low=-10, high=10, size=size)
    msg = r"bounds"
    with pytest.raises(np.AxisError, match=msg):
        num.squeeze(a, axis=3)


def test_array_axis_out_bound():
    size = (1, 2, 1)
    a = num.random.randint(-10, 10, size=size)
    msg = r"bounds"
    with pytest.raises(np.AxisError, match=msg):
        a.squeeze(axis=3)


@pytest.mark.parametrize("axes", (-1, -3))
def test_num_axis_negative(axes):
    size = (1, 2, 1)
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.squeeze(a, axis=axes)
    res_num = num.squeeze(b, axis=axes)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("axes", (-1, -3))
def test_array_axis_negative(axes):
    size = (1, 2, 1)
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = a.squeeze(axis=axes)
    res_num = b.squeeze(axis=axes)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_num_basic(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.squeeze(a)
    res_num = num.squeeze(b)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_array_basic(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = a.squeeze()
    res_num = b.squeeze()
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_num_axis(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    for xis in a.shape:
        if xis == 1:
            k = a.shape.index(xis)
            res_np = np.squeeze(a, axis=k)
            res_num = num.squeeze(b, axis=k)
            assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_array_axis(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)

    for xis in a.shape:
        if xis == 1:
            k = a.shape.index(xis)
            res_np = a.squeeze(axis=k)
            res_num = b.squeeze(axis=k)
            assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
