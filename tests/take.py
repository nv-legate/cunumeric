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
from test_tools.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def test():

    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    indices = mk_seq_array(np, (8,))
    indices_num = num.array(indices)

    # testing the case when no axis provided
    res = np.take(x, indices)
    res_num = num.take(x_num, indices_num)
    assert np.array_equal(res_num, res)

    # testing different modes with different axis
    res = np.take(x, indices, axis=0, mode="clip")
    res_num = num.take(x_num, indices_num, axis=0, mode="clip")
    assert np.array_equal(res_num, res)

    res = np.take(x, indices, axis=1, mode="wrap")
    res_num = num.take(x_num, indices_num, axis=1, mode="wrap")
    assert np.array_equal(res_num, res)

    res = np.take(x, indices, axis=2, mode="clip")
    res_num = num.take(x_num, indices_num, axis=2, mode="clip")
    assert np.array_equal(res_num, res)

    indices2 = mk_seq_array(np, (3,))
    indices2_num = num.array(indices2)

    res = np.take(x, indices2, axis=1)
    res_num = num.take(x_num, indices2_num, axis=1)
    assert np.array_equal(res_num, res)

    res = np.take(x, indices2, axis=2, mode="raise")
    res_num = num.take(x_num, indices2_num, axis=2, mode="raise")
    assert np.array_equal(res_num, res)

    # testing with output array
    out = np.ones((3, 4, 3), dtype=int)
    out_num = num.array(out)
    res = np.take(x, indices2, axis=2, mode="raise", out=out)
    res_num = num.take(x_num, indices2_num, axis=2, mode="raise", out=out_num)
    assert np.array_equal(out_num, out)

    # testing the case when indices is a scalar
    res = np.take(x, 7, axis=0, mode="clip")
    res_num = num.take(x_num, 7, axis=0, mode="clip")
    assert np.array_equal(res_num, res)

    res = np.take(x, 7, axis=0, mode="wrap")
    res_num = num.take(x_num, 7, axis=0, mode="wrap")
    assert np.array_equal(res_num, res)

    for ndim in range(1, LEGATE_MAX_DIM + 1):
        shape = (5,) * ndim
        np_arr = mk_seq_array(np, shape)
        num_arr = mk_seq_array(num, shape)
        np_indices = mk_seq_array(np, (4,))
        num_indices = mk_seq_array(num, (4,))
        res_np = np.take(np_arr, np_indices)
        res_num = num.take(num_arr, num_indices)
        assert np.array_equal(res_num, res_np)
        for axis in range(ndim):
            res_np = np.take(np_arr, np_indices, axis=axis)
            res_num = num.take(num_arr, num_indices, axis=axis)
            assert np.array_equal(res_num, res_np)
        np_indices = mk_seq_array(np, (8,))
        num_indices = mk_seq_array(num, (8,))
        res_np = np.take(np_arr, np_indices, mode="wrap")
        res_num = num.take(num_arr, num_indices, mode="wrap")
        assert np.array_equal(res_num, res_np)
        res_np = np.take(np_arr, np_indices, mode="clip")
        res_num = num.take(num_arr, num_indices, mode="clip")
        assert np.array_equal(res_num, res_np)
        for axis in range(ndim):
            res_np = np.take(np_arr, np_indices, axis=axis, mode="clip")
            res_num = num.take(num_arr, num_indices, axis=axis, mode="clip")
            assert np.array_equal(res_num, res_np)
            res_np = np.take(np_arr, np_indices, axis=axis, mode="wrap")
            res_num = num.take(num_arr, num_indices, axis=axis, mode="wrap")
            assert np.array_equal(res_num, res_np)

    return


if __name__ == "__main__":
    test()
