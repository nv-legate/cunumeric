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

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM
from utils.generators import mk_seq_array

import cunumeric as num

# cunumeric.clip(array, a_min, a_max, out=None, **kwargs)


@pytest.mark.xfail
def test_none_array():
    expected_exc = TypeError
    with pytest.raises(expected_exc):
        np.clip(None, a_min=0, a_max=0)
    with pytest.raises(expected_exc):
        # cunumeric raises
        # AttributeError: 'NoneType' object has no attribute 'clip'
        num.clip(None, a_min=0, a_max=0)


def test_empty_array():
    res_np = np.clip([0], a_min=0, a_max=0)
    res_num = num.clip([0], a_min=0, a_max=0)
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
def test_value_none():
    array = np.arange(0, 10)
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        # Numpy raises:
        # ValueError: One of max or min must be given
        np.clip(array, a_min=None, a_max=None)
    with pytest.raises(expected_exc):
        # cunumeric raises:
        # TypeError: int() argument must be a string, a bytes-like object
        # or a real number, not 'NoneType'
        num.clip(array, a_min=None, a_max=None)


@pytest.mark.xfail
def test_amin_amax():
    array = np.arange(0, 10)
    res_np = np.clip(array, a_min=9, a_max=5)
    res_num = num.clip(array, a_min=9, a_max=5)
    # the result is different
    # array = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # res_np = array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    # res_num = array([9, 9, 9, 9, 9, 9, 9, 9, 9, 5])
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
@pytest.mark.parametrize("amin", (-1, 0.5, 2.5, 5, 11))
def test_amin_value(amin):
    array = np.arange(0, 10)
    res_np = np.clip(array, a_min=amin, a_max=8.5)
    res_num = num.clip(array, a_min=amin, a_max=8.5)
    # res_np is not match res_num
    # in Numpy, if a_min or a_max is float, all data are marked as float,
    # while in cunumeric, all datas are int.
    # for example, amin = 5
    # array = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # res_np = array([5., 5., 5., 5., 5., 5., 6., 7., 8., 8.5])
    # res_num = array([5, 5, 5, 5, 5, 5, 6, 7, 8, 8])
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
def test_amin_complex():
    array = np.arange(0, 10)
    amin = 5 + 5j
    res_np = np.clip(array, a_min=amin, a_max=8.5)
    #  res_np = array([5. +5.j, 5. +5.j, 5. +5.j, 5. +5.j, 5. +5.j,
    #   5. +5.j, 6. +0.j, 7. +0.j, 8. +0.j, 8.5+0.j])
    res_num = num.clip(array, a_min=amin, a_max=8.5)
    # cunumeric raises:
    # TypeError: int() argument must be a string, a bytes-like object
    # or a real number, not 'complex'
    assert np.array_equal(res_np, res_num)


def test_value_list():
    array = np.arange(0, 5)
    amin = [2, 3, 4, 5, 1]
    amax = 8
    res_np = np.clip(array, a_min=amin, a_max=amax)
    res_num = num.clip(array, a_min=amin, a_max=amax)
    assert np.array_equal(res_np, res_num)


def test_value_list_negative():
    array = np.arange(0, 10)
    amin = [2, 3, 4, 5, 1]
    amax = 8
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.clip(array, a_min=amin, a_max=amax)
        # ValueError: operands could not be broadcast together
    with pytest.raises(expected_exc):
        num.clip(array, a_min=amin, a_max=amax)


def test_out_negative():
    array = np.arange(0, 5)
    out_a = np.arange(0, 3)
    amin = [2, 3, 4, 5, 1]
    amax = 8
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.clip(array, a_min=amin, a_max=amax, out=out_a)
    with pytest.raises(expected_exc):
        num.clip(array, a_min=amin, a_max=amax, out=out_a)


def test_out_negative_ndim():
    array = [[2, 3, 4], [3, 4, 5], [6, 6, 12]]
    np_arr = np.array(array)
    num_arr = num.array(array)
    out_a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    out_np = np.array(out_a)
    out_num = num.array(out_a)
    amin = 3
    amax = 8
    np.clip(np_arr, a_min=amin, a_max=amax, out=out_np)
    num.clip(num_arr, a_min=amin, a_max=amax, out=out_num)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_basic(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)

    amin = int(np.prod(shape) / 2)
    amax = np.prod(shape) - 1
    res_np = np.clip(np_arr, amin, amax)
    res_num = num.clip(num_arr, amin, amax)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_out(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)

    out_np = np.zeros(shape)
    out_num = num.zeros(shape)

    amin = int(np.prod(shape) / 2)
    amax = np.prod(shape) - 1
    np.clip(np_arr, amin, amax, out=out_np)
    num.clip(num_arr, amin, amax, out=out_num)
    assert np.array_equal(out_np, out_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
