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
from test_tools.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def test_3d():

    x = mk_seq_array(np, (256, 256, 100))
    x_num = mk_seq_array(num, (256, 256, 100))

    indices = mk_seq_array(np, (256, 256, 10)) % 100
    indices_num = num.array(indices)

    res = np.take_along_axis(x, indices, -1)
    res_num = num.take_along_axis(x_num, indices_num, -1)
    assert np.array_equal(res_num, res)


def test_None_axis():
    x = mk_seq_array(np, (256, 256, 100))
    x_num = mk_seq_array(num, (256, 256, 100))

    # testig the case when axis = None
    indices = mk_seq_array(np, (256,))
    indices_num = num.array(indices)

    res = np.take_along_axis(x, indices, None)
    res_num = num.take_along_axis(x_num, indices_num, None)
    assert np.array_equal(res_num, res)


N = 50


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (N,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_idx = (1,) * ndim
    np_indices = mk_seq_array(np, shape_idx) % N
    num_indices = mk_seq_array(num, shape_idx) % N
    for axis in range(ndim):
        res_np = np.take_along_axis(np_arr, np_indices, axis=axis)
        res_num = num.take_along_axis(num_arr, num_indices, axis=axis)
        assert np.array_equal(res_num, res_np)
    np_indices = mk_seq_array(np, (30,))
    num_indices = mk_seq_array(num, (30,))
    res_np = np.take_along_axis(np_arr, np_indices, None)
    res_num = num.take_along_axis(num_arr, num_indices, None)
    assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
