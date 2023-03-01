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
import scipy.signal as sig
from utils.comparisons import allclose

import cunumeric as num

SHAPES = [(100,), (10, 10), (10, 10, 10)]

FILTER_SHAPES = [(5,), (3, 5), (3, 5, 3)]


@pytest.mark.xfail
def test_none():
    # Numpy raises:
    # TypeError: unsupported operand type(s) for *: 'NoneType' and 'NoneType'
    with pytest.raises(AttributeError):
        num.convolve(None, None, mode="same")


def test_empty():
    msg = r"empty"
    with pytest.raises(ValueError, match=msg):
        num.convolve([], [], mode="same")


def test_diff_dims():
    shape1 = (5,) * 3
    shape2 = (5,) * 2
    arr1 = num.random.random(shape1)
    arr2 = num.random.random(shape2)
    with pytest.raises(RuntimeError):
        num.convolve(arr1, arr2, mode="same")


def check_convolve(a, v):
    anp = a.__array__()
    vnp = v.__array__()

    out = num.convolve(a, v, mode="same")
    if a.ndim > 1:
        out_np = sig.convolve(anp, vnp, mode="same")
    else:
        out_np = np.convolve(anp, vnp, mode="same")

    assert allclose(out, out_np)


@pytest.mark.parametrize(
    "shape, filter_shape", zip(SHAPES, FILTER_SHAPES), ids=str
)
def test_double(shape, filter_shape):
    a = num.random.rand(*shape)
    v = num.random.rand(*filter_shape)

    check_convolve(a, v)
    check_convolve(v, a)


@pytest.mark.parametrize(
    "shape, filter_shape", zip(SHAPES, FILTER_SHAPES), ids=str
)
def test_int(shape, filter_shape):
    a = num.random.randint(0, 5, shape)
    v = num.random.randint(0, 5, filter_shape)

    check_convolve(a, v)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
