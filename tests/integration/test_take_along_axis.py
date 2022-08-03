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

N = 10


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (N,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_idx = (1,) * ndim
    np_indices = mk_seq_array(np, shape_idx) % N
    num_indices = mk_seq_array(num, shape_idx) % N
    for axis in range(-1, ndim):
        res_np = np.take_along_axis(np_arr, np_indices, axis=axis)
        res_num = num.take_along_axis(num_arr, num_indices, axis=axis)
        assert np.array_equal(res_num, res_np)
    np_indices = mk_seq_array(np, (3,))
    num_indices = mk_seq_array(num, (3,))
    res_np = np.take_along_axis(np_arr, np_indices, None)
    res_num = num.take_along_axis(num_arr, num_indices, None)
    assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
