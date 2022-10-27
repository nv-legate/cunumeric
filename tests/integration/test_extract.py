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
from utils.generators import mk_seq_array

import cunumeric as num

DIM = 5
SIZES = [
    (0,),
    1,
    5,
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

VALUES = [
    [0],
    [42],
    [42 + 3j],
    [11, 12, 13],
    [True, False, False, True],
    [42.3, 42.3, 42.3, 42.3, 42.3],
    [np.inf, np.Inf],
]


@pytest.mark.xfail
def test_none_array():
    res_np = np.extract([0], None)  # return []
    res_num = num.extract(
        [0], None
    )  # AttributeError: 'NoneType' object has no attribute 'size'
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
def test_empty_array():
    res_np = np.extract([0], [])  # return []
    res_num = num.extract(
        [0], []
    )  # ValueError: arr array and condition array must be of same size
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
def test_none_condition():
    a = num.array([1, 2, 3, 4])
    res_np = np.extract(None, a)  # all return []
    res_num = num.extract(
        None, a
    )  # AttributeError: 'NoneType' object has no attribute 'size'
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize(
    "con", (-3, 0, 3, False, True, [2], [2, 3], [2, -3, 4], [1, 2, 3, 4, 5])
)
def test_negative_condition(con):
    a = num.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        num.extract(con, a)


@pytest.mark.parametrize("shape", SIZES, ids=str)
def test_extract_basic(shape):
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    # make sure condition is between 1 and 2
    np_condition = mk_seq_array(np, shape) % 2
    num_condition = mk_seq_array(num, shape) % 2

    res_np = np.extract(np_condition, np_arr)
    res_num = num.extract(num_condition, num_arr)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("shape", SIZES, ids=str)
@pytest.mark.parametrize("vals", VALUES, ids=str)
def test_place_basic(shape, vals):
    arr_np = mk_seq_array(np, shape)
    arr_num = num.array(mk_seq_array(num, shape))

    condition_np = mk_seq_array(np, shape) % 3
    condition_num = mk_seq_array(num, shape) % 3

    vals_np = np.array(vals).astype(arr_np.dtype)
    vals_num = num.array(vals_np)

    np.place(arr_np, condition_np, vals_np)
    num.place(arr_num, condition_num, vals_num)

    assert np.array_equal(arr_np, arr_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
