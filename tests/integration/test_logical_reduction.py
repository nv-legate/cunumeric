import numpy as np
import pytest
from utils.comparisons import allclose

import cunumeric as num


def test_logical_and_reduce():
    input = [[[12, 0, 1, 2], [9, 0, 0, 1]], [[0, 0, 0, 5], [1, 1, 1, 1]]]
    in_num = num.array(input)
    in_np = np.array(input)

    axes = [None, 0, 1, 2, (0, 1, 2)]
    for axis in axes:
        out_num = num.logical_and.reduce(in_num, axis=axis)
        out_np = np.logical_and.reduce(in_np, axis=axis)
        assert allclose(out_num, out_np, check_dtype=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
