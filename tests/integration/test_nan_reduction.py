from math import prod

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM

import cunumeric as num

NAN_FUNCS = ("nanmax", "nanmin", "nanprod", "nansum")

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

    @pytest.mark.parametrize("func_name", NAN_FUNCS)
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

        # relax criteria when doing floating ops with nans excluded
        if func_np.__name__ == "nanprod" or func_np.__name__ == "nansum":
            assert np.allclose(out_num, out_np, rtol=1e-4)
        else:
            assert np.array_equal(out_num, out_np)


class TestCornerCases:
    """
    These are corner cases
    """

    def test_empty_for_min_max(self, func_name):
        with pytest.raises(ValueError):
            num.nanmin([])
        with pytest.raises(ValueError):
            num.nanmax([])

    def test_empty_for_prod_sum(self):
        assert num.nanprod([]) == 1.0
        assert num.nansum([]) == 0.0


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
