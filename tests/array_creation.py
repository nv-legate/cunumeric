# Copyright 2021 NVIDIA Corporation
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

import cunumeric as num


def test():
    x = num.array([1, 2, 3])
    y = np.array([1, 2, 3])
    z = num.array(y)
    assert np.array_equal(x, z)
    assert x.dtype == z.dtype

    x = num.array([1, 2, 3])
    y = num.array(x)
    assert num.array_equal(x, y)
    assert x.dtype == y.dtype

    xe = num.empty((2, 3))
    ye = np.empty((2, 3))
    assert xe.shape == ye.shape
    assert xe.dtype == ye.dtype

    xz = num.zeros((2, 3))
    yz = np.zeros((2, 3))
    assert np.array_equal(xz, yz)
    assert xz.dtype == yz.dtype

    xo = num.ones((2, 3))
    yo = np.ones((2, 3))
    assert np.array_equal(xo, yo)
    assert xo.dtype == yo.dtype

    xf = num.full((2, 3), 3)
    yf = np.full((2, 3), 3)
    assert np.array_equal(xf, yf)
    assert xf.dtype == yf.dtype

    xel = num.empty_like(x)
    yel = np.empty_like(y)
    assert xel.shape == yel.shape
    assert xel.dtype == yel.dtype

    xzl = num.zeros_like(x)
    yzl = np.zeros_like(y)
    assert np.array_equal(xzl, yzl)
    assert xzl.dtype == yzl.dtype

    xol = num.ones_like(x)
    yol = np.ones_like(y)
    assert np.array_equal(xol, yol)
    assert xol.dtype == yol.dtype

    xfl = num.full_like(x, 3)
    yfl = np.full_like(y, 3)
    assert np.array_equal(xfl, yfl)
    assert xfl.dtype == yfl.dtype

    x = num.arange(1)
    y = np.arange(1)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    x = num.arange(10)
    y = np.arange(10)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    x = num.arange(10, dtype=np.int32)
    y = np.arange(10, dtype=np.int32)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    x = num.arange(2.0, 10.0)
    y = np.arange(2.0, 10.0)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    x = num.arange(2, 30, 3)
    y = np.arange(2, 30, 3)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    # xfls = num.full_like(x, '3', dtype=np.str_)
    # yfls = np.full_like(y, '3', dtype=np.str_)
    # assert(num.array_equal(xfls, yfls))
    # assert(xfls.dtype == yfls.dtype)

    return


if __name__ == "__main__":
    test()
