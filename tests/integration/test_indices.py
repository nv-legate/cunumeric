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

import numpy as np
import pytest

import cunumeric as cn
from legate.core import LEGATE_MAX_DIM


class TestIndicesErrors:
    """
    this class is to test negative cases
    indices(dimensions, dtype=<class 'int'>, sparse=False)
    """

    def test_int_dimensions(self):
        dimensions = 3
        msg = r"'int' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            cn.indices(dimensions)

    def test_negative_dimensions(self):
        dimensions = -3
        msg = r"'int' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            cn.indices(dimensions)

    def test_float_dimensions(self):
        dimensions = 3.2
        msg = r"'float' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            cn.indices(dimensions)

    def test_negative_tuple_dimensions(self):
        dimensions = (1, -1)
        # numpy raises: "ValueError: negative dimensions are not allowed"
        msg = r"Invalid shape"
        with pytest.raises(ValueError, match=msg):
            cn.indices(dimensions)

    def test_float_tuple_dimensions(self):
        dimensions = (3.5, 2.5)
        # numpy raises:
        # "TypeError: 'float' object cannot be interpreted as an integer"
        msg = r"expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            cn.indices(dimensions)


class TestIndices:
    """
    These are positive cases compared with numpy
    """

    @pytest.mark.parametrize("dimensions", [(0,), (0, 0), (0, 1), (1, 1)])
    def test_indices_zero(self, dimensions):
        np_res = np.indices(dimensions)
        cn_res = cn.indices(dimensions)

        assert np.array_equal(np_res, cn_res)

    @pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM))
    def test_indices_basic(self, ndim):
        dimensions = tuple(random.randint(1, 5) for i in range(ndim))

        np_res = np.indices(dimensions)
        cn_res = cn.indices(dimensions)
        assert np.array_equal(np_res, cn_res)

    @pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM))
    def test_indices_dtype_none(self, ndim):
        dimensions = tuple(random.randint(1, 5) for i in range(ndim))

        np_res = np.indices(dimensions, dtype=None)
        cn_res = cn.indices(dimensions, dtype=None)
        assert np.array_equal(np_res, cn_res)

    @pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM))
    def test_indices_dtype_float(self, ndim):
        dimensions = tuple(random.randint(1, 5) for i in range(ndim))
        np_res = np.indices(dimensions, dtype=float)
        cn_res = cn.indices(dimensions, dtype=float)
        assert np.array_equal(np_res, cn_res)

    @pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM))
    def test_indices_sparse(self, ndim):
        dimensions = tuple(random.randint(1, 5) for i in range(ndim))
        np_res = np.indices(dimensions, sparse=True)
        cn_res = cn.indices(dimensions, sparse=True)
        for i in range(len(np_res)):
            assert np.array_equal(np_res[i], cn_res[i])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
