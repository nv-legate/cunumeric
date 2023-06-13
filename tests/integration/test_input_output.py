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

import os
import pickle

import numpy as np
import pytest
from utils.generators import mk_0to1_array, mk_seq_array

import cunumeric as num


def test_ndarray_dumps():
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    dump_np = arr_np.dumps()
    dump_num = arr_num.dumps()
    assert np.allclose(pickle.loads(dump_np), arr_np)
    assert np.allclose(pickle.loads(dump_num), arr_num)


@pytest.mark.parametrize("order", ["C", "F", "A"], ids=str)
def test_ndarray_tobytes(order):
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    bytes_np = arr_np.tobytes(order)
    bytes_num = arr_num.tobytes(order)
    assert bytes_np == bytes_num


@pytest.mark.parametrize("shape", [(3, 2, 4), []], ids=str)
def test_ndarray_tolist(shape):
    arr_np = mk_0to1_array(np, shape)
    arr_num = mk_0to1_array(num, shape)
    list_np = arr_np.tolist()
    list_num = arr_num.tolist()
    assert list_np == list_num


@pytest.mark.parametrize("order", ["C", "F", "A"], ids=str)
def test_ndarray_tostring(order):
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    str_np = arr_np.tostring(order)
    str_num = arr_num.tostring(order)
    assert str_np == str_num


def test_ndarray_dump():
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    dump_np = "npdump"
    arr_np.dump(file=dump_np)
    dump_num = "numdump"
    arr_num.dump(file=dump_num)
    with open(dump_np, "rb") as f:
        assert np.allclose(pickle.load(f), arr_np)
    with open(dump_num, "rb") as f:
        assert np.allclose(pickle.load(f), arr_num)
    os.remove(dump_np)
    os.remove(dump_num)


def test_ndarray_tofile():
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    dump_np = "npdump"
    arr_np.tofile(dump_np)
    dump_num = "numdump"
    arr_num.tofile(dump_num)
    with open(dump_np, "rb") as file_np, open(dump_num, "rb") as file_num:
        assert file_np.readlines() == file_num.readlines()
    os.remove(dump_np)
    os.remove(dump_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
