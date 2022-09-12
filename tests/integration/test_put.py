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
from legate.core import LEGATE_MAX_DIM


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


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    np_indices = mk_seq_array(np, (4 * ndim,))
    num_indices = mk_seq_array(num, (4 * ndim,))
    shape_val = (3,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10

    np.put(np_arr, np_indices, np_values)
    num.put(num_arr, num_indices, num_values)
    assert np.array_equal(np_arr, num_arr)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("mode", ("wrap", "clip"))
def test_ndim_mode(ndim, mode):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_in = (3,) * ndim
    np_indices = mk_seq_array(np, shape_in) * 2
    num_indices = mk_seq_array(num, shape_in) * 2
    shape_val = (2,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10

    np.put(np_arr, np_indices, np_values, mode=mode)
    num.put(num_arr, num_indices, num_values, mode=mode)
    assert np.array_equal(np_arr, num_arr)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
