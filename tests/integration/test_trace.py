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

import random
from itertools import permutations

import numpy as np
import pytest
from test_tools.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def test_2d():
    a = np.arange(8).reshape((2, 4))
    a_num = num.array(a)
    res = np.trace(a)
    res_num = num.trace(a_num)
    assert np.array_equal(res, res_num)

    res = np.trace(a, dtype=float)
    res_num = num.trace(a_num, dtype=float)
    assert np.array_equal(res, res_num)


def test_3d():
    a = np.arange(8).reshape((2, 2, 2))
    a_num = num.array(a)
    res = np.trace(a)
    res_num = num.trace(a_num)
    assert np.array_equal(res, res_num)

    res = np.trace(a, offset=1)
    res_num = num.trace(a_num, offset=1)
    assert np.array_equal(res, res_num)

    res = np.trace(a, offset=1, axis1=1, axis2=2)
    res_num = num.trace(a_num, offset=1, axis1=1, axis2=2)
    assert np.array_equal(res, res_num)

    out = np.array([1, 2], dtype=float)
    out_num = num.array(out)
    np.trace(a, out=out)
    num.trace(a_num, out=out_num)
    assert np.array_equal(out, out_num)

    np.trace(a, dtype=int, out=out)
    num.trace(a_num, dtype=int, out=out_num)
    assert np.array_equal(out, out_num)


def test_4d():
    a = np.arange(24).reshape((2, 2, 2, 3))
    a_num = num.array(a)
    res = np.trace(a)
    res_num = num.trace(a_num)
    assert np.array_equal(res, res_num)


@pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    a_shape = tuple(random.randint(1, 9) for i in range(ndim))
    np_array = mk_seq_array(np, a_shape)
    num_array = mk_seq_array(num, a_shape)

    # test trace
    for axes in permutations(range(ndim), 2):
        diag_size = min(a_shape[axes[0]], a_shape[axes[1]]) - 1
        for offset in range(-diag_size + 1, diag_size):
            assert np.array_equal(
                np_array.trace(offset, axes[0], axes[1]),
                num_array.trace(offset, axes[0], axes[1]),
            )


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
