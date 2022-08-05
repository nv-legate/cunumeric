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
from legate.core import LEGATE_MAX_DIM
from utils.comparisons import allclose
from utils.generators import mk_0to1_array

import cunumeric as cn

VECTOR_ORDS = [None, np.inf, -np.inf, 0, 1, -1, 2, -2]

# TODO: Add "nuc", 2, -2 once they are implemented
MATRIX_ORDS = [None, "fro", np.inf, -np.inf, 1, -1]

np_arrays = [
    mk_0to1_array(np, (3,) * ndim) - 0.5
    for ndim in range(0, LEGATE_MAX_DIM + 1)
]
cn_arrays = [
    mk_0to1_array(cn, (3,) * ndim) - 0.5
    for ndim in range(0, LEGATE_MAX_DIM + 1)
]


@pytest.mark.parametrize("ord", VECTOR_ORDS)
@pytest.mark.parametrize("keepdims", [False, True])
def test_noaxis_1d(ord, keepdims):
    np_res = np.linalg.norm(np_arrays[1], ord=ord, keepdims=keepdims)
    cn_res = cn.linalg.norm(cn_arrays[1], ord=ord, keepdims=keepdims)
    assert allclose(np_res, cn_res)


@pytest.mark.parametrize("ord", MATRIX_ORDS)
@pytest.mark.parametrize("keepdims", [False, True])
def test_noaxis_2d(ord, keepdims):
    np_res = np.linalg.norm(np_arrays[2], ord=ord, keepdims=keepdims)
    cn_res = cn.linalg.norm(cn_arrays[2], ord=ord, keepdims=keepdims)
    assert allclose(np_res, cn_res)


@pytest.mark.parametrize("ndim", [0] + list(range(3, LEGATE_MAX_DIM + 1)))
@pytest.mark.parametrize("keepdims", [False, True])
def test_noaxis_other(ndim, keepdims):
    np_res = np.linalg.norm(np_arrays[ndim], keepdims=keepdims)
    cn_res = cn.linalg.norm(cn_arrays[ndim], keepdims=keepdims)
    assert allclose(np_res, cn_res)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("ord", VECTOR_ORDS)
@pytest.mark.parametrize("keepdims", [False, True])
def test_axis_1d(ndim, ord, keepdims):
    np_res = np.linalg.norm(
        np_arrays[ndim], ord=ord, axis=0, keepdims=keepdims
    )
    cn_res = cn.linalg.norm(
        cn_arrays[ndim], ord=ord, axis=0, keepdims=keepdims
    )
    assert allclose(np_res, cn_res)


@pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("ord", MATRIX_ORDS)
@pytest.mark.parametrize("keepdims", [False, True])
def test_axis_2d(ndim, ord, keepdims):
    np_res = np.linalg.norm(
        np_arrays[ndim], ord=ord, axis=(0, 1), keepdims=keepdims
    )
    cn_res = cn.linalg.norm(
        cn_arrays[ndim], ord=ord, axis=(0, 1), keepdims=keepdims
    )
    assert allclose(np_res, cn_res)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
