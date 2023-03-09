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


def test_dtype_negative():
    arr = num.arange(5, dtype=float)
    msg = r"integer type"
    with pytest.raises(TypeError, match=msg):
        num.bincount(arr)


def test_weight_mismatch():
    v_num = num.random.randint(0, 9, size=N)
    w_num = num.random.randn(N + 1)
    msg = r"same shape"
    with pytest.raises(ValueError, match=msg):
        num.bincount(v_num, weights=w_num)


def test_out_size():
    arr = num.array([0, 1, 1, 3, 2, 1, 7, 23])
    assert num.bincount(arr).size == num.amax(arr) + 1


@pytest.mark.skip()
def test_array_ndim():
    size = (2,) * 3
    arr = num.random.randint(0, high=9, size=size)
    # Numpy raises : ValueError: object too deep for desired array
    # cuNumeric run aborted
    with pytest.raises(ValueError):
        num.bincount(arr)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("minlength", MINLENGTHS)
def test_bincount_basic(dtype, minlength):
    v_num = num.random.randint(0, MAX_VAL, size=N, dtype=dtype)
    out_num = num.bincount(v_num, minlength=minlength)

    v_np = v_num.__array__()
    out_np = np.bincount(v_np, minlength=minlength)
    assert num.array_equal(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_weights(dtype):
    v_num = num.random.randint(0, MAX_VAL, size=N, dtype=dtype)
    w_num = num.random.randn(N)
    out_num = num.bincount(v_num, weights=w_num)

    v_np = v_num.__array__()
    w_np = w_num.__array__()
    out_np = np.bincount(v_np, weights=w_np)

    assert allclose(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_high_bins(dtype):
    v_num = num.array([0, LARGE_NUM_BINS], dtype=dtype)
    out_num = num.bincount(v_num)

    v_np = v_num.__array__()
    out_np = np.bincount(v_np)

    assert num.array_equal(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_weights_high_bins(dtype):
    v_num = num.array([0, LARGE_NUM_BINS], dtype=dtype)
    w_num = num.random.randn(2)
    out_num = num.bincount(v_num, weights=w_num)

    v_np = v_num.__array__()
    w_np = w_num.__array__()
    out_np = np.bincount(v_np, weights=w_np)

    assert allclose(out_np, out_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
