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
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, DIM - 1, DIM),
    (2, DIM - 1, DIM),
    (DIM, DIM, DIM),
]


@pytest.mark.xfail
def test_none_array_compare():
    res_num = num.transpose(None)  # AttributeError: 'NoneType'
    res_np = np.transpose(None)  # return None
    assert np.array_equal(res_num, res_np, equal_nan=True)


def test_none_array():
    # numpy returned None
    msg = r"NoneType"
    with pytest.raises(AttributeError, match=msg):
        num.transpose(None)


@pytest.mark.parametrize("axes", ((1, 1, 1), (1, 2, 3), (1, 2), (1, 2, 0, 1)))
def test_num_invalid_axis(axes):
    size = (2, 3, 4)
    a = num.random.randint(low=-10, high=10, size=size)
    with pytest.raises(ValueError):
        num.transpose(a, axes=axes)


def test_num_int_axis():
    size = (2, 3, 4)
    a = num.random.randint(low=-10, high=10, size=size)
    # numpy raises "ValueError: axes don't match array"
    # cunumeric raises "TypeError"
    with pytest.raises(TypeError):
        num.transpose(a, axes=2)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_num_basic(size):
    a = np.random.randint(low=-10, high=10, size=size)
    res_np = np.transpose(a)
    res_num = num.transpose(a)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_array_basic(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = a.transpose()
    res_num = b.transpose()
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_num_round(size):
    a = num.random.randint(low=-10, high=10, size=size)
    b = num.transpose(a)
    c = num.transpose(b)
    assert num.array_equal(c, a)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_array_round(size):
    a = num.random.randint(low=-10, high=10, size=size)
    b = a.transpose()
    c = b.transpose()
    assert num.array_equal(c, a)


@pytest.mark.parametrize(
    "size",
    (
        (0,),
        (1),
        (DIM),
    ),
)
@pytest.mark.parametrize("axes", (-3, 0, 3, (1,), (3, 1)))
def test_num_axes_1d(size, axes):
    # when array.dim==1, it returns the array itself,
    # no matter what the axes value is
    a = num.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.transpose(a, axes=axes)
    res_num = num.transpose(b, axes=axes)
    assert num.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "size",
    ((1, 0), (1, 1), (1, DIM), (DIM, 1), (DIM - 1, DIM - 2), (DIM, DIM)),
)
@pytest.mark.parametrize("axes", ((0, 1), (1, 0)))
def test_num_axes_2d(size, axes):
    a = num.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.transpose(a, axes=axes)
    res_num = num.transpose(b, axes=axes)
    assert num.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "size",
    (
        (1, 0, 1),
        (1, 1, 1),
        (DIM, DIM - 1, 1),
        (1, 1, DIM),
        (2, 3, 4),
        (DIM, DIM, DIM),
    ),
)
@pytest.mark.parametrize(
    "axes", ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
)
def test_num_axes_3d(size, axes):
    a = num.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.transpose(a, axes=axes)
    res_num = num.transpose(b, axes=axes)
    assert num.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "size",
    (
        (0,),
        (1),
        (DIM),
    ),
)
@pytest.mark.parametrize("axes", (-3, 0, 3, (1,), (3, 1)))
def test_array_axes_1d(size, axes):
    # when array.dim==1, it returns the array itself,
    # no matter what the axes value is
    a = num.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = a.transpose(axes=axes)
    res_num = b.transpose(axes=axes)
    assert num.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "size",
    ((1, 0), (1, 1), (1, DIM), (DIM, 1), (DIM - 1, DIM - 2), (DIM, DIM)),
)
@pytest.mark.parametrize("axes", ((0, 1), (1, 0)))
def test_array_axes_2d(size, axes):
    a = num.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = a.transpose(axes=axes)
    res_num = b.transpose(axes=axes)
    assert num.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "size",
    (
        (1, 0, 1),
        (1, 1, 1),
        (DIM, DIM - 1, 1),
        (1, 1, DIM),
        (2, 3, 4),
        (DIM, DIM, DIM),
    ),
)
@pytest.mark.parametrize(
    "axes", ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
)
def test_array_axes_3d(size, axes):
    a = num.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = a.transpose(axes=axes)
    res_num = b.transpose(axes=axes)
    assert num.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
