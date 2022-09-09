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
from utils.generators import mk_seq_array

import cunumeric as num

# from legate.core import LEGATE_MAX_DIM


def test_1d():
    x_np = np.arange(10)
    x_num = num.arange(10)
    i_np = np.array([1, 3, 5])
    i_num = num.array(i_np)
    v_np = np.array([100])
    v_num = num.array(v_np)
    np.put(x_np, i_np, v_np)
    num.put(x_num, i_num, v_num)
    assert np.array_equal(x_num, x_np)


def test_raise():

    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    indices = mk_seq_array(np, (8,))
    indices_num = num.array(indices)
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)

    np.put(x, indices, values)
    num.put(x_num, indices_num, values_num)
    assert np.array_equal(x_num, x)


# def test_modes():
#
#    x = mk_seq_array(np, (3, 4, 5))
#    x_num = mk_seq_array(num, (3, 4, 5))
#    indices = mk_seq_array(np, (8,))*2
#    indices_num = num.array(indices)
#    values = mk_seq_array(np, (6,))*10
#    values_num = num.array(values)
#
#    np.put(x, indices, values, mode="clip")
#    num.put(x_num, indices_num, values, mode="clip")
#    assert np.array_equal(x_num, x)
#
#    np.put(x, indices, values, mode="wrap")
#    num.put(x_num, indices_num, values, mode="wrap")
#    assert np.array_equal(x_num, x)
#
# def test_scalar():
#    # testing the case when indices is a scalar
#    x = mk_seq_array(np, (3, 4, 5))
#    x_num = mk_seq_array(num, (3, 4, 5))
#    values = mk_seq_array(np, (6,))*10
#    values_num = num.array(values)
#
#    np.put(x, 0, values)
#    num.put(x_num, 0, values)
#    assert np.array_equal(x_num, x)
#
#    np.put(x, 1, -10, mode)
#    num.put(x_num, 1, -10, mode)
#    assert np.array_equal(x_num, x)
#
#
# def test_nd_indices():
#    x = mk_seq_array(np, (15))
#    x_num = mk_seq_array(num, (15))
#    indices = mk_seq_array(np, (3,2))*2
#    indices_num = num.array(indices)
#    values = mk_seq_array(np, (2,2))*10
#    values_num = num.array(values)
#
#    np.put(x, indices, values, mode)
#    num.put(x_num, indices_num, values, mode)
#    assert np.array_equal(x_num, x)
#
# @pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
# def test_ndim(ndim):
#    shape = (5,) * ndim
#    np_arr = mk_seq_array(np, shape)
#    num_arr = mk_seq_array(num, shape)
#    np_indices = mk_seq_array(np, (4,))
#    num_indices = mk_seq_array(num, (4,))
#    np_values = mk_seq_array(np, (2,))*10
#    num_values = mk_seq_array(num, (2,))*10
#
#    np.put(np_arr, np_indices, np_values)
#    num.put(num_arr, num_indices, num_values)
#    assert np.array_equal(np_arr, num_arr)
#
#    np_indices = mk_seq_array(np, (8,))
#    num_indices = mk_seq_array(num, (8,))
#    np.put(np_arr, np_indices, np_values, mode="wrap")
#    num.put(num_arr, num_indices,num_values, mode="wrap")
#    assert np.array_equal(np_arr, num_arr)
#
#    np_arr = mk_seq_array(np, shape)
#    num_arr = mk_seq_array(num, shape)
#    np_arr.put(np_indices,np_values, mode="clip")
#    num_arr.put(num_indices, num_values, mode="clip")
#    assert np.array_equal(np_arr, num_arr)
#
#    return

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
