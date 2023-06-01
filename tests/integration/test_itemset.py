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
import random

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM

import cunumeric as num


@pytest.mark.xfail
def test_no_itemset():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)

    expected_exc = ValueError
    with pytest.raises(expected_exc):
        arr_num.itemset()
        # Numpy raises ValueError: itemset must have
        # at least one argument
    with pytest.raises(expected_exc):
        arr_np.itemset()
        # cuNumeric raises KeyError: 'itemset() requires
        # at least one argument'


@pytest.mark.xfail
def test_invalid_itemset():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)

    expected_exc = ValueError
    with pytest.raises(expected_exc):
        arr_np.itemset(8)
        # Numpy raises ValueError: can only convert an array of size 1
        # to a Python scalar
    with pytest.raises(expected_exc):
        arr_num.itemset(8)
        # cuNumeric raises KeyError: 'invalid key'


@pytest.mark.xfail
def test_out_of_index():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)
    expected_exc = IndexError
    with pytest.raises(expected_exc):
        arr_np.itemset(10, 4)
        # Numpy raises IndexError: index 10 is out of bounds for size 9
    with pytest.raises(expected_exc):
        arr_num.itemset(10, 4)
        # cuNumeric set the value of index 1 as 4
        # Original array:
        # [[193 212 238]
        #  [ 97 103 225]
        #  [197 107 174]]
        # new array
        # [[193   4 238]
        #  [ 97 103 225]
        #  [197 107 174]]


@pytest.mark.xfail
def test_tuple_out_of_index():
    shape = (3, 3)
    arr_num = num.random.randint(0, 300, size=shape)
    arr_np = np.array(arr_num)

    expected_exc = IndexError
    with pytest.raises(expected_exc):
        arr_np.itemset((2, 3), 4)
        # Numpy raises IndexError: index 3 is out of bounds
        # for axis 1 with size 3
    with pytest.raises(expected_exc):
        arr_num.itemset((2, 2), 4)
        # cuNumeric raises ValueError: Out-of-bounds projection on
        # dimension 0 with index 3 for a store of shape Shape((3,))


def generate_itemset(ndim):
    index_list = []
    max_index = pow(4, ndim) - 1
    random_index = random.randint(0, max_index)
    index_list.append(random_index)
    index_list.append(max_index)
    random_tuple = []
    for i in range(0, ndim):
        random_x = random.randint(0, 3)
        random_tuple.append(random_x)
    index_list.append(tuple(random_tuple))
    return index_list


@pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (4,) * ndim
    arr_num = num.random.randint(0, 30, size=shape)
    arr_np = np.array(arr_num)
    for itemset in generate_itemset(ndim):
        arr_num_copy = arr_num
        arr_np_copy = arr_np
        arr_num_copy.itemset(itemset, 40)
        arr_np_copy.itemset(itemset, 40)

        assert np.array_equal(arr_np_copy, arr_num_copy)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
