from math import prod

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM

import cunumeric as num

NAN_ARG_FUNCS = ("nanargmax",)

NDIMS = range(LEGATE_MAX_DIM + 1)

DISALLOWED_DTYPES = (
    np.complex64,
    np.complex128,
    np.complex256,
)


class Testbasic:
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic(self, func_name, ndim, keepdims):
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

        out_np = np.unravel_index(func_np(in_np, keepdims=keepdims), shape)
        out_num = np.unravel_index(func_num(in_num, keepdims=keepdims), shape)

        index_array_np = in_np[out_np]
        index_array_num = in_num[out_num]

        assert np.array_equal(out_num, out_np)
        assert np.array_equal(index_array_num, index_array_np)

    # def test_out(self,):
    #    pass

    # test different datatypes


class TestXFail:
    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("disallowed_dtype", DISALLOWED_DTYPES)
    def test_disallowed_dtypes(self, func_name, ndim, disallowed_dtype):
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
        shape = (3,) * ndim
        in_num = num.empty(shape)
        in_num.fill(num.nan)

        func_num = getattr(num, func_name)

        msg = r"Array/Slice contains only NaNs"
        with pytest.raises(ValueError, match=msg):
            func_num(in_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    def test_slice_nan(self, func_name, ndim):
        shape = (3,) * ndim
        in_num = num.empty(shape)
        in_num.fill(num.nan)

        if ndim == 1:
            in_num[:] = np.nan
        elif ndim == 2:
            in_num[:, 0] = np.nan
        else:
            in_num[0, ...] = np.nan

        func_num = getattr(num, func_name)
        func_num(in_num)

        msg = r"Array/Slice contains only NaNs"
        with pytest.raises(ValueError, match=msg):
            func_num(in_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    def test_out_mismatch(self, func_name, ndim):
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
