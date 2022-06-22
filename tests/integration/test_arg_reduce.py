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

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


@pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("keepdims", [True, False])
def test_argmax_and_argmin(ndim, keepdims):
    shape = (5,) * ndim

    in_np = np.random.random(shape)
    in_num = num.array(in_np)

    for fn in ("argmax", "argmin"):
        fn_np = getattr(np, fn)
        fn_num = getattr(num, fn)
        assert np.array_equal(
            fn_np(in_np, keepdims=keepdims), fn_num(in_num, keepdims=keepdims)
        )
        if in_num.ndim == 1:
            continue
        for axis in range(in_num.ndim):
            out_np = fn_np(in_np, axis=axis, keepdims=keepdims)
            out_num = fn_num(in_num, axis=axis, keepdims=keepdims)
            assert np.array_equal(out_np, out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
