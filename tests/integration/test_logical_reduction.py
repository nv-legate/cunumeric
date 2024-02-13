import numpy as np
import pytest

import cunumeric as num


@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 1, 2)])
def test_logical_reductions(axis):
    input = [[[12, 0, 1, 2], [9, 0, 0, 1]], [[0, 0, 0, 5], [1, 1, 1, 1]]]
    in_num = num.array(input)
    in_np = np.array(input)

    out_num = num.logical_and.reduce(in_num, axis=axis)
    out_np = np.logical_and.reduce(in_np, axis=axis)
    assert num.array_equal(out_num, out_np)

    out_num = num.logical_or.reduce(in_num, axis=axis)
    out_np = np.logical_or.reduce(in_np, axis=axis)
    assert num.array_equal(out_num, out_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
