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


def test_None():

    x = mk_seq_array(np, (256,))
    x_num = mk_seq_array(num, (256,))

    indices = mk_seq_array(np, (125,))
    indices_num = num.array(indices)

    np.put_along_axis(x, indices, -10, None)
    num.put_along_axis(x_num, indices_num, -10, None)
    assert np.array_equal(x_num, x)


N = 10


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (N,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = num.array(np_arr)

    shape_idx = (1,) * ndim
    np_indices = mk_seq_array(np, shape_idx) % N
    num_indices = mk_seq_array(num, shape_idx) % N
    for axis in range(-1, ndim):
        np_a = np_arr.copy()
        num_a = num_arr.copy()
        np.put_along_axis(np_a, np_indices, 8, axis=axis)
        num.put_along_axis(num_a, num_indices, 8, axis=axis)
        assert np.array_equal(np_a, num_a)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
