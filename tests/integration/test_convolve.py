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
UNSUPPORTED_MODES = ["valid", "full"]
UNSUPPORTED_NDIM = [4, 5]


@pytest.mark.xfail
def test_none():
    expected_exc = TypeError
    with pytest.raises(expected_exc):
        num.convolve(None, None, mode="same")
        # cuNumeric raises AttributeError
    with pytest.raises(expected_exc):
        np.convolve(None, None, mode="same")


def test_empty():
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        num.convolve([], [], mode="same")
    with pytest.raises(expected_exc):
        np.convolve([], [], mode="same")


def test_diff_dims():
    shape1 = (5,) * 3
    shape2 = (5,) * 2
    arr1 = num.random.random(shape1)
    arr2 = num.random.random(shape2)
    expected_exc = RuntimeError
    with pytest.raises(expected_exc):
        num.convolve(arr1, arr2, mode="same")
    with pytest.raises(expected_exc):
        np.convolve(arr1, arr2, mode="same")


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


def test_dtype():
    shape = (5,) * 2
    arr1 = num.random.randint(0, 5, shape)
    arr2 = num.random.random(shape)
    out_num = num.convolve(arr1, arr2, mode="same")
    out_np = np.convolve(arr1, arr2, mode="same")
    assert allclose(out_num, out_np)


@pytest.mark.xfail
@pytest.mark.parametrize("mode", UNSUPPORTED_MODES)
def test_modes(mode):
    shape = (5,) * 2
    arr1 = num.random.random(shape)
    arr2 = num.random.random(shape)
    out_num = num.convolve(arr1, arr2, mode=mode)
    # when mode!="same", cunumeric raises
    # NotImplementedError: Need to implement other convolution modes
    out_np = np.convolve(arr1, arr2, mode=mode)
    assert allclose(out_num, out_np)


@pytest.mark.xfail
@pytest.mark.parametrize("ndim", UNSUPPORTED_NDIM)
def test_ndim(ndim):
    shape = (5,) * ndim
    arr1 = num.random.random(shape)
    arr2 = num.random.random(shape)
    out_num = num.convolve(arr1, arr2, mode="same")
    # cunumeric raises,  NotImplementedError: 4-D arrays are not yet supported
    out_np = np.convolve(arr1, arr2, mode="same")
    assert allclose(out_num, out_np)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
