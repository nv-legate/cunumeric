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


def test_array():
    x = num.array([1, 2, 3])
    y = np.array([1, 2, 3])
    z = num.array(y)
    assert np.array_equal(x, z)
    assert x.dtype == z.dtype

    x = num.array([1, 2, 3])
    y = num.array(x)
    assert num.array_equal(x, y)
    assert x.dtype == y.dtype


def test_empty():
    xe = num.empty((2, 3))
    ye = np.empty((2, 3))
    assert xe.shape == ye.shape
    assert xe.dtype == ye.dtype


def test_zeros():
    xz = num.zeros((2, 3))
    yz = np.zeros((2, 3))
    assert np.array_equal(xz, yz)
    assert xz.dtype == yz.dtype


def test_ones():
    xo = num.ones((2, 3))
    yo = np.ones((2, 3))
    assert np.array_equal(xo, yo)
    assert xo.dtype == yo.dtype


def test_full():
    xf = num.full((2, 3), 3)
    yf = np.full((2, 3), 3)
    assert np.array_equal(xf, yf)
    assert xf.dtype == yf.dtype


def test_empty_like():
    x = num.array([1, 2, 3])
    y = num.array(x)
    xel = num.empty_like(x)
    yel = np.empty_like(y)
    assert xel.shape == yel.shape
    assert xel.dtype == yel.dtype


def test_zeros_like():
    x = num.array([1, 2, 3])
    y = num.array(x)
    xzl = num.zeros_like(x)
    yzl = np.zeros_like(y)
    assert np.array_equal(xzl, yzl)
    assert xzl.dtype == yzl.dtype


def test_ones_like():
    x = num.array([1, 2, 3])
    y = num.array(x)
    xol = num.ones_like(x)
    yol = np.ones_like(y)
    assert np.array_equal(xol, yol)
    assert xol.dtype == yol.dtype


def test_full_like():
    x = num.array([1, 2, 3])
    y = num.array(x)
    xfl = num.full_like(x, 3)
    yfl = np.full_like(y, 3)
    assert np.array_equal(xfl, yfl)
    assert xfl.dtype == yfl.dtype

    # xfls = num.full_like(x, '3', dtype=np.str_)
    # yfls = np.full_like(y, '3', dtype=np.str_)
    # assert(num.array_equal(xfls, yfls))
    # assert(xfls.dtype == yfls.dtype)


ARANGE_ARGS = [
    (1,),
    (10,),
    (2.0, 10.0),
    (2, 30, 3),
]


@pytest.mark.parametrize("args", ARANGE_ARGS, ids=str)
def test_arange(args):
    x = num.arange(*args)
    y = np.arange(*args)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype


def test_arange_with_dtype():
    x = num.arange(10, dtype=np.int32)
    y = np.arange(10, dtype=np.int32)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
