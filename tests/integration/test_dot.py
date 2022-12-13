# Copyright 2021-2022 NVIDIA Corporation
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
from utils.contractions import check_default
from utils.generators import mk_0to1_array

import cunumeric as num
from cunumeric.utils import dot_modes


@pytest.mark.parametrize("b_ndim", range(LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("a_ndim", range(LEGATE_MAX_DIM + 1))
def test_dot(a_ndim, b_ndim):
    name = f"dot({a_ndim} x {b_ndim})"
    modes = dot_modes(a_ndim, b_ndim)

    def operation(lib, *args, **kwargs):
        return lib.dot(*args, **kwargs)

    check_default(name, modes, operation)


class TestDotErrors:
    def setup_method(self):
        self.A = mk_0to1_array(num, (5, 3))
        self.B = mk_0to1_array(num, (3, 2))

    @pytest.mark.parametrize(
        "shapeA",
        ((3,), (4, 3), (5, 4, 3)),
        ids=lambda shapeA: f"(shapeA={shapeA})",
    )
    def test_a_b_invalid_shape(self, shapeA):
        A = mk_0to1_array(num, shapeA)
        B = mk_0to1_array(num, (2, 2))
        with pytest.raises(ValueError):
            num.dot(A, B)

    @pytest.mark.parametrize(
        "shape", ((5,), (2,), (5, 3)), ids=lambda shape: f"(shape={shape})"
    )
    def test_out_invalid_shape(self, shape):
        out = num.zeros(shape)
        with pytest.raises(ValueError):
            num.dot(self.A, self.B, out=out)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "dtype", (np.float32, np.int64), ids=lambda dtype: f"(dtype={dtype})"
    )
    def test_out_invalid_dtype(self, dtype):
        # In Numpy, for np.float32 and np.int64, it raises ValueError
        # In cuNumeric,
        # for np.float32, it pass
        # for np.int64, it raises TypeError: Unsupported type: int64
        out = np.zeros((5, 2), dtype=dtype)
        with pytest.raises(ValueError):
            np.dot(self.A, self.B, out=out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
