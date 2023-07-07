# Copyright 2022-2023 NVIDIA Corporation
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

from math import prod

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM

import cunumeric as num

NAN_ARG_FUNCS = ("nanargmax", "nanargmin")

NDIMS = range(LEGATE_MAX_DIM + 1)

DISALLOWED_DTYPES = (
    np.complex64,
    np.complex128,
)

# Note that when an element is repeated mulitple times in an array,
# the output from cuNumeric and numpy will vary. This is expected and
# is not a bug. So, whenever we compare with numpy, we try to make
# sure the elements in the array are unique. Another way to circumvent
# this problem would be to make sure that argument corresponding
# to the max/min is indeed the max/min element in the array


class TestNanArgReductions:
    """
    These are positive cases compared with numpy
    """

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic(self, func_name, ndim, keepdims):
        """This test inserts a NaN in the array and checks if the
        output from cuNumeric and numpy match
        """
        shape = (5,) * ndim
        size = prod(shape)
        in_np = np.random.random(shape)

        # set an element to nan
        index_nan = np.random.randint(low=0, high=size)
        index_nan = np.unravel_index(index_nan, shape)
        in_num = num.array(in_np)
        in_num[index_nan] = num.nan
        in_np[index_nan] = np.nan

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        # make sure numpy and cunumeric give the same out array and max val
        out_np = np.unravel_index(func_np(in_np, keepdims=keepdims), shape)
        out_num = np.unravel_index(func_num(in_num, keepdims=keepdims), shape)

        index_array_np = in_np[out_np]
        index_array_num = in_num[out_num]

        assert np.array_equal(out_num, out_np)
        assert np.array_equal(index_array_num, index_array_np)

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    def test_out(self, func_name, ndim):
        """This test checks that the out argument is updated with the
        output"""

        shape = (3,) * ndim
        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        func_num = getattr(num, func_name)
        func_np = getattr(np, func_name)

        axes = list(range(0, ndim - 1))
        for axis in axes:
            _shape = list(shape)
            _shape[axis] = 1

            out_num = num.empty(_shape, dtype=int)
            func_num(in_num, out=out_num, axis=axis, keepdims=True)

            out_np = np.empty(_shape, dtype=int)
            func_np(in_np, out=out_np, axis=axis, keepdims=True)

            assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_floating_point_types(self, func_name, ndim, dtype):
        """This test checks the most frequently used datatypes
        to make sure that the results match with numpy. The
        arrays may or may not contain NaNs.
        """
        shape = (4,) * ndim
        size = prod(shape)

        in_np = np.random.choice(size, size, replace=False) * np.random.rand(1)
        in_np = in_np.astype(dtype)

        in_num = num.array(in_np, dtype=dtype)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        out_np = func_np(in_np)
        out_num = func_num(in_num)

        assert np.array_equal(out_num, out_np)


class TestXFail:
    """
    This class is to test negative cases
    """

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("disallowed_dtype", DISALLOWED_DTYPES)
    def test_disallowed_dtypes(self, func_name, ndim, disallowed_dtype):
        """This test checks if we raise an error for types that are
        disallowed."""
        shape = (2,) * ndim
        in_num = num.random.random(shape).astype(disallowed_dtype)

        func_num = getattr(num, func_name)

        msg = r"operation is not supported for complex-type arrays"
        with pytest.raises(ValueError, match=msg):
            func_num(in_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", NDIMS)
    def test_all_nan(self, func_name, ndim):
        """This test checks if we comply with the expected behavior when
        the array contains only NaNs. The expected behavior is to
        raise a ValueError.
        """
        shape = (3,) * ndim
        in_num = num.zeros(shape)
        in_num.fill(num.nan)

        func_num = getattr(num, func_name)
        func_num(in_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
    def test_slice_nan(self, func_name, ndim):
        """This test checks if we comply with the expected behavior when
        a slice contains only NaNs. The expected behavior is to raise a
        ValueError.
        """
        shape = (3,) * ndim
        in_num = num.random.rand(shape)

        if ndim == 2:
            in_num[:, 0] = num.nan
        else:
            in_num[0, ...] = num.nan

        func_num = getattr(num, func_name)
        func_num(in_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    def test_out_mismatch(self, func_name):
        """This test checks if we raise an error when the output shape is
        incorrect"""

        ndim = 2
        shape = (3,) * ndim
        func_num = getattr(num, func_name)

        in_num = np.random.random(shape)

        # shape
        index_array = np.random.randint(low=1, high=4, size=(2, 2))
        func_num(in_num, out=index_array)

        # dtype
        index_array = np.random.rand(2, 2)
        func_num(in_num, out=index_array)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
