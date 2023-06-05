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

    @pytest.mark.parametrize("func_name", NAN_FUNCS)
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic(self, func_name, ndim, keepdims):
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

        # relax criteria when doing floating ops with nans excluded
        if func_np.__name__ == "nanprod" or func_np.__name__ == "nansum":
            assert np.allclose(out_num, out_np, rtol=1e-4)
        else:
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

    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
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
        with pytest.raises(ValueError):
            func_num(in_num)

    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    @pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
    def test_slice_nan(self, func_name, ndim):
        """This test checks if we comply with the expected behavior when
        a slice contains only NaNs. The expected behavior is to issue a
        RuntimeWarning.
        """
        shape = (3,) * ndim

        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        if ndim == 2:
            in_num[0, :] = num.nan
            in_np[0, :] = np.nan
        else:
            in_num[0, ...] = num.nan
            in_np[0, ...] = np.nan

        func_num = getattr(num, func_name)
        func_np = getattr(np, func_name)
        func_num(in_num)

        with pytest.warns(RuntimeWarning):
            out_np = func_np(in_np, axis=1)
        with pytest.warns(RuntimeWarning):
            out_num = func_num(in_num, axis=1)

        assert np.array_equal(out_np, out_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
