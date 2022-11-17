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

import cunumeric as num

np_arr = np.eye(4)
np_vec = np.arange(4).astype(np.float64)
num_arr = num.array(np_arr)
num_vec = num.array(np_vec)
indices = [0, 3, 1, 2]


def test_array_function_implemented():
    res_np = np.dot(np_arr, np_vec)
    res_num = np.dot(num_arr, num_vec)
    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, num.ndarray)  # implemented


def test_array_function_unimplemented():
    res_np = np.linalg.tensorsolve(np_arr, np_vec)
    res_num = np.linalg.tensorsolve(num_arr, num_vec)
    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, np.ndarray)  # unimplemented


def test_array_ufunc_through_array_op():
    assert np.array_equal(num_vec + num_vec, np_vec + np_vec)
    assert isinstance(num_vec + np_vec, num.ndarray)
    assert isinstance(np_vec + num_vec, num.ndarray)


def test_array_ufunc_call():
    res_np = np.add(np_vec, np_vec)
    res_num = np.add(num_vec, num_vec)
    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, num.ndarray)  # implemented


def test_array_ufunc_reduce():
    res_np = np.add.reduce(np_vec)
    res_num = np.add.reduce(num_vec)
    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, num.ndarray)  # implemented


def test_array_ufunc_accumulate():
    res_np = np.add.accumulate(np_vec)
    res_num = np.add.accumulate(num_vec)
    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, np.ndarray)  # unimplemented


def test_array_ufunc_reduceat():
    res_np = np.add.reduceat(np_vec, indices)
    res_num = np.add.reduceat(num_vec, indices)
    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, np.ndarray)  # unimplemented


def test_array_ufunc_outer():
    res_np = np.add.outer(np_vec, np_vec)
    res_num = np.add.outer(num_vec, num_vec)
    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, np.ndarray)  # unimplemented


def test_array_ufunc_at():
    res_np = np.full((4,), 42)
    res_num = num.full((4,), 42)
    np.add.at(res_np, indices, np_vec)
    np.add.at(res_num, indices, num_vec)
    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, num.ndarray)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
