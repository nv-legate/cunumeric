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

import cunumeric as num


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("dtype", ("B", "i", "?"))
@pytest.mark.parametrize("bitorder", ("little", "big"))
def test_packbits(ndim, dtype, bitorder):
    in_np = np.array([], dtype=dtype)
    in_num = num.array([], dtype=dtype)
    out_np = np.packbits(in_np, bitorder=bitorder)
    out_num = num.packbits(in_num, bitorder=bitorder)
    assert np.array_equal(out_np, out_num)

    for extent in (3, 5, 8, 16):
        shape = (extent,) * ndim
        in_np = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
        in_num = num.array(in_np)

        out_np = np.packbits(in_np, bitorder=bitorder)
        out_num = num.packbits(in_num, bitorder=bitorder)
        assert np.array_equal(out_np, out_num)

        for axis in range(ndim):
            out_np = np.packbits(in_np, axis=axis, bitorder=bitorder)
            out_num = num.packbits(in_num, axis=axis, bitorder=bitorder)
            assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("bitorder", ("little", "big"))
def test_unpackbits(ndim, bitorder):
    in_np = np.array([], dtype="B")
    in_num = num.array([], dtype="B")
    out_np = np.unpackbits(in_np, bitorder=bitorder)
    out_num = num.unpackbits(in_num, bitorder=bitorder)
    assert np.array_equal(out_np, out_num)

    for extent in (3, 5, 8, 16):
        shape = (extent,) * ndim
        in_np = np.random.randint(low=0, high=255, size=shape, dtype="B")
        in_num = num.array(in_np)

        out_np = np.unpackbits(in_np, bitorder=bitorder)
        out_num = num.unpackbits(in_num, bitorder=bitorder)
        assert np.array_equal(out_np, out_num)

        out_np = np.unpackbits(in_np, count=extent // 2, bitorder=bitorder)
        out_num = num.unpackbits(in_num, count=extent // 2, bitorder=bitorder)
        assert np.array_equal(out_np, out_num)

        for axis in range(ndim):
            out_np = np.unpackbits(in_np, axis=axis, bitorder=bitorder)
            out_num = num.unpackbits(in_num, axis=axis, bitorder=bitorder)
            assert np.array_equal(out_np, out_num)

            out_np = np.unpackbits(
                in_np, count=extent // 2, axis=axis, bitorder=bitorder
            )
            out_num = num.unpackbits(
                in_num, count=extent // 2, axis=axis, bitorder=bitorder
            )
            assert np.array_equal(out_np, out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
