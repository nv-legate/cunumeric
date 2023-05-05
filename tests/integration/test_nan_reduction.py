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
    np.complex256,
)

ALLOWED_DTYPES = (
    np.float16,
    np.float32,
    np.float64,
)


class TestNanReductions:
    """
    These are positive cases compared with numpy
    """

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

        out_np = func_np(in_np, keepdims=keepdims)
        out_num = func_num(in_num, keepdims=keepdims)

        assert np.array_equal(out_num, out_np)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
