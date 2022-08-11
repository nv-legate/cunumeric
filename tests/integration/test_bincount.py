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
from utils.comparisons import allclose

import cunumeric as num

N = 8000
MAX_VAL = 9
LARGE_NUM_BINS = 20000

DTYPES = [np.int64, np.int32, np.int16]
MINLENGTHS = [0, 5, 15]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("minlength", MINLENGTHS)
def test_bincount_basic(dtype, minlength):
    v_num = num.random.randint(0, MAX_VAL, size=N, dtype=dtype)

    v_np = v_num.__array__()

    out_np = np.bincount(v_np, minlength=minlength)
    out_num = num.bincount(v_num, minlength=minlength)
    assert num.array_equal(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_weights(dtype):
    v_num = num.random.randint(0, MAX_VAL, size=N, dtype=dtype)
    w_num = num.random.randn(N)

    v_np = v_num.__array__()
    w_np = w_num.__array__()

    out_np = np.bincount(v_np, weights=w_np)
    out_num = num.bincount(v_num, weights=w_num)
    assert allclose(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_high_bins(dtype):
    v_num = num.array([0, LARGE_NUM_BINS], dtype=dtype)

    v_np = v_num.__array__()

    out_np = np.bincount(v_np)
    out_num = num.bincount(v_num)
    assert num.array_equal(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_weights_high_bins(dtype):
    v_num = num.array([0, LARGE_NUM_BINS], dtype=dtype)
    w_num = num.random.randn(2)

    v_np = v_num.__array__()
    w_np = w_num.__array__()

    out_np = np.bincount(v_np, weights=w_np)
    out_num = num.bincount(v_num, weights=w_num)
    assert allclose(out_np, out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
