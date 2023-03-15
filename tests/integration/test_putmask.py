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


def test_scalar():
    x = mk_seq_array(np, (3,))
    x_num = mk_seq_array(num, (3,))
    values = np.zeros((), dtype=np.int32)
    values_num = num.zeros((), dtype=np.int32)
    mask = (x % 2).astype(bool)
    mask_num = num.array(mask)
    np.putmask(x[:1], mask[2:], values)
    num.putmask(x_num[:1], mask_num[2:], values_num)
    assert np.array_equal(x_num, x)

    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    mask = (x % 2).astype(bool)
    mask_num = num.array(mask)
    np.putmask(x, mask, 100)
    num.putmask(x_num, mask_num, 100)
    assert np.array_equal(x_num, x)

    x = np.zeros((), dtype=np.int32)
    x_num = num.zeros((), dtype=np.int32)
    mask = False
    mask_num = False
    np.putmask(x, mask, -1)
    num.putmask(x_num, mask_num, -1)
    assert np.array_equal(x_num, x)

    x = np.zeros((), dtype=np.int32)
    x_num = num.zeros((), dtype=np.int32)
    mask = True
    mask_num = True
    np.putmask(x, mask, -1)
    num.putmask(x_num, mask_num, -1)
    assert np.array_equal(x_num, x)

    # testing the case when indices is a scalar
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)
    mask = (x % 2).astype(bool)
    mask_num = num.array(mask)
    np.putmask(x, mask, values[:1])
    num.putmask(x_num, mask_num, values_num[:1])
    assert np.array_equal(x_num, x)


def test_type_convert():
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)
    mask = x % 2
    mask_num = x_num % 2
    np.putmask(x, mask, values)
    num.putmask(x_num, mask_num, values_num)
    assert np.array_equal(x_num, x)

    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)
    mask = np.zeros(
        (
            3,
            4,
            5,
        )
    )
    mask_num = num.zeros((3, 4, 5))
    np.putmask(x, mask, values)
    num.putmask(x_num, mask_num, values_num)
    assert np.array_equal(x_num, x)

    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    x = x.astype(np.int32)
    x_num = x_num.astype(np.int32)
    mask = np.zeros(
        (
            3,
            4,
            5,
        )
    )
    mask_num = num.zeros((3, 4, 5))
    np.putmask(x, mask, 11)
    num.putmask(x_num, mask_num, 11)
    assert np.array_equal(x_num, x)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    np_mask = (np_arr % 2).astype(bool)
    num_mask = (num_arr % 2).astype(bool)
    # scalar_val
    np.putmask(np_arr, np_mask, -10)
    num.putmask(num_arr, num_mask, -10)
    assert np.array_equal(np_arr, num_arr)

    # val is the same shape:
    np_val = np_arr * 10
    num_val = num_arr * 10
    np.putmask(np_arr, np_mask, np_val)
    num.putmask(num_arr, num_mask, num_val)
    assert np.array_equal(np_arr, num_arr)

    # val is different shape, but the same size
    shape_val = (np_arr.size,)
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10
    np.putmask(np_arr, np_mask, np_values)
    num.putmask(num_arr, num_mask, num_values)
    assert np.array_equal(np_arr, num_arr)

    # val is different shape and different size for vals and array
    shape_val = (2,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10
    np.putmask(np_arr, np_mask, np_values)
    num.putmask(num_arr, num_mask, num_values)
    assert np.array_equal(np_arr, num_arr)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
