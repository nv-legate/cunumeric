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


@pytest.mark.parametrize("mode", ("wrap", "clip"))
def test_scalar(mode):
    # testing the case when indices is a scalar
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)

    np.put(x, 0, values)
    num.put(x_num, 0, values_num)
    assert np.array_equal(x_num, x)

    np.put(x, 1, -10, mode)
    num.put(x_num, 1, -10, mode)
    assert np.array_equal(x_num, x)

    # checking transformed array
    y = x[:1]
    y_num = x_num[:1]
    np.put(y, 0, values)
    num.put(y_num, 0, values_num)
    assert np.array_equal(x_num, x)


def test_indices_type_convert():
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)
    indices = np.array([1, 2], dtype=np.int32)
    indices_num = num.array(indices)
    np.put(x, indices, values)
    num.put(x_num, indices_num, values_num)
    assert np.array_equal(x_num, x)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_in = (3,) * ndim
    np_indices = mk_seq_array(np, shape_in)
    num_indices = mk_seq_array(num, shape_in)
    shape_val = (2,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10

    np.put(np_arr, np_indices, np_values)
    num.put(num_arr, num_indices, num_values)
    assert np.array_equal(np_arr, num_arr)


INDICES = ([1, 2, 3, 100], [[2, 1], [3, 100]], [1], [100])


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("mode", ("wrap", "clip"))
@pytest.mark.parametrize("indices", INDICES)
def test_ndim_mode(ndim, mode, indices):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_val = (2,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10

    np.put(np_arr, indices, np_values, mode=mode)
    num.put(num_arr, indices, num_values, mode=mode)
    assert np.array_equal(np_arr, num_arr)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
