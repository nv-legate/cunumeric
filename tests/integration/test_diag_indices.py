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

import cunumeric as cn


def test_diag_indices_default_ndim():
    np_res = np.diag_indices(10)
    cn_res = cn.diag_indices(10)
    assert np.array_equal(np_res, cn_res)


@pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM + 1))
def test_diag_indices_basic(ndim):
    np_res = np.diag_indices(10, ndim)
    cn_res = cn.diag_indices(10, ndim)
    assert np.array_equal(np_res, cn_res)


@pytest.mark.parametrize("n", [0, 0.0, 1, 10.5])
@pytest.mark.parametrize("ndim", [-4, 0, 1])
def test_diag_indices(n, ndim):
    np_res = np.diag_indices(n, ndim)
    cn_res = cn.diag_indices(n, ndim)
    assert np.array_equal(np_res, cn_res)


class TestDiagIndicesErrors:
    @pytest.mark.parametrize("n", [-10.5, -1])
    def test_negative_n(self, n):
        msg = "Invalid shape"
        print("[robinw]: test-1")
        with pytest.raises(ValueError, match=msg):
            cn.diag_indices(n)

    @pytest.mark.xfail
    @pytest.mark.parametrize("n", [-10.5, -1])
    def test_negative_n_compare_with_numpy(self, n):
        # np.diag_indices(-10.5) returns empty 2-D array, dtype=float64
        # np.diag_indices(-1) returns empty 2-D array, dtype=int32
        # cn.diag_indices(-10.5) raises ValueError
        # cn.diag_indices(-1) raises ValueError
        np_res = np.diag_indices(n)
        cn_res = cn.diag_indices(n)
        assert np.array_equal(np_res, cn_res)

    def test_none_n(self):
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            cn.diag_indices(None)

    @pytest.mark.parametrize("ndim", [-1.5, 0.0, 1.5])
    def test_float_ndim(self, ndim):
        msg = "can't multiply sequence by non-int of type 'float'"
        with pytest.raises(TypeError, match=msg):
            cn.diag_indices(10, ndim)

    def test_none_ndim(self):
        msg = "can't multiply sequence by non-int of type 'NoneType'"
        with pytest.raises(TypeError, match=msg):
            cn.diag_indices(10, None)


@pytest.mark.parametrize("size", [(5,), (0,)], ids=str)
@pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
def test_diag_indices_from_basic(size, ndim):
    shape = size * ndim
    a = np.ones(shape, dtype=int)
    a_cn = cn.array(a)
    np_res = np.diag_indices_from(a)
    cn_res = cn.diag_indices_from(a_cn)
    assert np.array_equal(np_res, cn_res)


class TestDiagIndicesFromErrors:
    @pytest.mark.parametrize("size", [(5,), (0,)], ids=str)
    def test_1d(self, size):
        a = np.ones(size, dtype=int)
        msg = "input array must be at least 2-d"
        with pytest.raises(ValueError, match=msg):
            cn.diag_indices_from(a)

    def test_0d(self):
        a = np.array(3)
        msg = "input array must be at least 2-d"
        with pytest.raises(ValueError, match=msg):
            cn.diag_indices_from(a)

    @pytest.mark.parametrize(
        "size",
        [
            (
                5,
                2,
            ),
            (5, 0, 0),
        ],
        ids=str,
    )
    def test_unequal_length(self, size):
        a = np.ones(size, dtype=int)
        msg = "All dimensions of input must be of equal length"
        with pytest.raises(ValueError, match=msg):
            cn.diag_indices_from(a)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
