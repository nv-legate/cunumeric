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

NAN_FUNCS = ("nanmax", "nanmin", "nanprod", "nansum")

NDIMS = range(LEGATE_MAX_DIM + 1)


class TestNanReductions:
    """
    These are positive cases compared with numpy
    """

    @pytest.mark.parametrize("func_name", ("nansum", "nanprod"))
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic_nan_sum_prod(self, func_name, ndim, keepdims):
        """This test sets an element to NaN and checks if the output
        from cuNumeric and numpy match."""
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

        out_np = func_np(in_np, keepdims=keepdims)
        out_num = func_num(in_num, keepdims=keepdims)

        assert np.allclose(out_num, out_np, rtol=1e-4)

    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic_nan_min_max(self, func_name, ndim, keepdims):
        """This test sets an element to NaN and checks if the output
        from cuNumeric and numpy match."""
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

        out_np = func_np(in_np, keepdims=keepdims)
        out_num = func_num(in_num, keepdims=keepdims)

        assert np.array_equal(out_num, out_np)

    @pytest.mark.parametrize("func_name", NAN_FUNCS)
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

            out_num = num.empty(_shape)
            func_num(in_num, out=out_num, axis=axis, keepdims=True)

            out_np = np.empty(_shape)
            func_np(in_np, out=out_np, axis=axis, keepdims=True)

            assert np.allclose(out_num, out_np, rtol=1e-4)

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_complex_dtype_nansum(self, ndim, dtype, keepdims):
        """This test checks if nansum works as expected for complex
        datatypes. The parametrized datatype is that of real/complex
        component, so float32 would correspond to complex64 and
        float64 would correspond to complex128.
        """
        shape = (3,) * ndim
        size = prod(shape)

        r = np.random.random(shape).astype(dtype)
        c = np.random.random(shape).astype(dtype)

        # get index of nan
        index_nan = np.random.randint(low=0, high=size)
        index_nan = np.unravel_index(index_nan, shape)

        # set an element to nan
        r[index_nan] = np.nan
        c[index_nan] = np.nan

        in_np = r + 1j * c

        in_num = num.array(in_np)
        in_num[index_nan] = num.nan

        out_num = num.nansum(in_num, keepdims=keepdims)
        out_np = np.nansum(in_np, keepdims=keepdims)

        assert np.allclose(out_num, out_np, rtol=1e-4)

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_complex_dtype_nanprod(self, ndim, keepdims):
        """This test checks if nanprod works as expected for complex
        datatypes. The parametrized datatype is that of real/complex
        component, so float32 would correspond to complex64.
        """
        shape = (3,) * ndim
        size = prod(shape)

        dtype = np.float32

        r = np.random.random(shape).astype(dtype)
        c = np.random.random(shape).astype(dtype)

        # get index of nan
        index_nan = np.random.randint(low=0, high=size)
        index_nan = np.unravel_index(index_nan, shape)

        # set just the real component to nan
        r[index_nan] = np.nan

        # set an element to nan
        in_np = r + 1j * c

        in_num = num.array(in_np)

        out_num = num.nanprod(in_num, keepdims=keepdims)
        out_np = np.nanprod(in_np, keepdims=keepdims)

        assert np.allclose(out_num, out_np, rtol=1e-4)

    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    def test_slice_nan_min_max(self, func_name):
        """This test checks if nanmin and nanmax return nan for
        a slice that contains all-NaNs
        """
        shape = (3, 3)
        in_num = num.random.random(shape)

        in_num[:, 0] = num.nan
        func_num = getattr(num, func_name)
        out_num = func_num(in_num, axis=0)

        assert num.any(num.isnan(out_num))

    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    def test_all_nans_nan_min_max(self, ndim, func_name):
        shape = (3,) * ndim
        in_num = num.random.random(shape)
        in_num[...] = np.nan

        func_num = getattr(num, func_name)
        out_num = func_num(in_num)

        assert num.isnan(out_num)

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    def test_all_nans_nanprod(self, ndim):
        shape = (3,) * ndim
        in_num = num.random.random(shape)
        in_num[...] = np.nan

        out_num = num.nanprod(in_num)

        assert out_num == 1.0

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    def test_all_nans_nansum(self, ndim):
        shape = (3,) * ndim
        in_num = num.random.random(shape)
        in_num[...] = np.nan

        out_num = num.nansum(in_num)

        assert out_num == 0.0


class TestCornerCases:
    """
    These are corner cases where we check with empty arrays
    """

    def test_empty_for_min_max(self):
        with pytest.raises(ValueError):
            num.nanmin([])
        with pytest.raises(ValueError):
            num.nanmax([])

    def test_empty_for_prod_sum(self):
        assert num.nanprod([]) == 1.0
        assert num.nansum([]) == 0.0


class TestXFail:
    """
    This class is to test negative cases
    """

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("disallowed_dtype", (np.complex64, np.complex128))
    def test_disallowed_dtypes(self, func_name, ndim, disallowed_dtype):
        """This test checks if we raise an error for types that are
        disallowed."""
        shape = (3,) * ndim
        in_num = num.random.random(shape).astype(disallowed_dtype)

        func_num = getattr(num, func_name)

        msg = r"operation is not supported for complex64 and complex128 types"
        with pytest.raises(NotImplementedError, match=msg):
            func_num(in_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    def test_disallowed_dtype_nan_min_max(self, func_name):
        ndim = 1
        shape = (3,) * ndim
        dtype = np.float32

        r = num.random.random(shape).astype(dtype)
        c = num.random.random(shape).astype(dtype)
        r[0] = c[0] = num.nan
        in_num = r + 1j * c

        func_num = getattr(num, func_name)
        with pytest.raises(NotImplementedError):
            func_num(in_num)

    @pytest.mark.xfail
    def test_disallowed_dtype_nanprod(self):
        ndim = 1
        shape = (3,) * ndim
        dtype = np.float64

        r = num.random.random(shape).astype(dtype)
        c = num.random.random(shape).astype(dtype)
        r[0] = c[0] = num.nan
        in_num = r + 1j * c

        with pytest.raises(NotImplementedError):
            num.nanprod(in_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
