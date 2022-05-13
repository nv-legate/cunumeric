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

import cunumeric as cn


def test_array_dunders():
    np_arr = np.eye(4)
    np_vec = np.arange(4).astype(np.float64)
    cn_arr = cn.array(np_arr)
    cn_vec = cn.array(np_vec)
    indices = [0, 3, 1, 2]

    # module-level functions

    np_res = np.dot(np_arr, np_vec)
    cn_res = np.dot(cn_arr, cn_vec)
    assert np.array_equal(np_res, cn_res)
    assert isinstance(cn_res, cn.ndarray)  # implemented

    np_res = np.linalg.solve(np_arr, np_vec)
    cn_res = np.linalg.solve(cn_arr, cn_vec)
    assert np.array_equal(np_res, cn_res)
    assert isinstance(cn_res, np.ndarray)  # unimplemented

    # ufunc invoked indirectly, through __op__ array method

    assert np.array_equal(cn_vec + cn_vec, np_vec + np_vec)
    assert isinstance(cn_vec + np_vec, cn.ndarray)
    assert isinstance(np_vec + cn_vec, cn.ndarray)

    # ufunc invoked directly, on NumPy ufunc object

    np_res = np.add(np_vec, np_vec)
    cn_res = np.add(cn_vec, cn_vec)
    assert np.array_equal(np_res, cn_res)
    assert isinstance(cn_res, cn.ndarray)  # implemented

    np_res = np.add.reduce(np_vec)
    cn_res = np.add.reduce(cn_vec)
    assert np.array_equal(np_res, cn_res)
    assert isinstance(cn_res, cn.ndarray)  # implemented

    np_res = np.add.accumulate(np_vec)
    cn_res = np.add.accumulate(cn_vec)
    assert np.array_equal(np_res, cn_res)
    assert isinstance(cn_res, np.ndarray)  # unimplemented

    np_res = np.add.reduceat(np_vec, indices)
    cn_res = np.add.reduceat(cn_vec, indices)
    assert np.array_equal(np_res, cn_res)
    assert isinstance(cn_res, np.ndarray)  # unimplemented

    np_res = np.add.outer(np_vec, np_vec)
    cn_res = np.add.outer(cn_vec, cn_vec)
    assert np.array_equal(np_res, cn_res)
    assert isinstance(cn_res, np.ndarray)  # unimplemented

    np.add.at(np_vec, indices, np_vec)
    np.add.at(cn_vec, indices, cn_vec)
    assert np.array_equal(np_vec, cn_vec)
    assert isinstance(cn_vec, cn.ndarray)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
