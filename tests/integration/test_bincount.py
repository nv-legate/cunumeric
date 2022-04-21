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

N = 8000

DTYPES = [np.int64, np.int32, np.int16]


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_basic(dtype):
    v_num = num.random.randint(0, 9, size=N, dtype=dtype)

    v_np = v_num.__array__()

    out_np = np.bincount(v_np)
    out_num = num.bincount(v_num)
    assert num.array_equal(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_weights(dtype):
    v_num = num.random.randint(0, 9, size=N, dtype=dtype)
    w_num = num.random.randn(N)

    v_np = v_num.__array__()
    w_np = w_num.__array__()

    out_np = np.bincount(v_np, weights=w_np)
    out_num = num.bincount(v_num, weights=w_num)
    assert num.allclose(out_np, out_num)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
