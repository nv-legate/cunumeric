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

import legate.numpy as lg


def test():
    x = lg.array([1, 2, 3])
    y = np.array([1, 2, 3])
    z = lg.array(y)
    assert np.array_equal(x, z)
    assert x.dtype == z.dtype

    xe = lg.empty((2, 3))
    ye = np.empty((2, 3))
    assert lg.shape(xe) == np.shape(ye)
    assert xe.dtype == ye.dtype

    xz = lg.zeros((2, 3))
    yz = np.zeros((2, 3))
    assert np.array_equal(xz, yz)
    assert xz.dtype == yz.dtype

    xo = lg.ones((2, 3))
    yo = np.ones((2, 3))
    assert np.array_equal(xo, yo)
    assert xo.dtype == yo.dtype

    xf = lg.full((2, 3), 3)
    yf = np.full((2, 3), 3)
    assert np.array_equal(xf, yf)
    assert xf.dtype == yf.dtype

    xel = lg.empty_like(x)
    yel = np.empty_like(y)
    assert lg.shape(xel) == np.shape(yel)
    assert xel.dtype == yel.dtype

    xzl = lg.zeros_like(x)
    yzl = np.zeros_like(y)
    assert np.array_equal(xzl, yzl)
    assert xzl.dtype == yzl.dtype

    xol = lg.ones_like(x)
    yol = np.ones_like(y)
    assert np.array_equal(xol, yol)
    assert xol.dtype == yol.dtype

    xfl = lg.full_like(x, 3)
    yfl = np.full_like(y, 3)
    assert np.array_equal(xfl, yfl)
    assert xfl.dtype == yfl.dtype

    x = lg.arange(10)
    y = np.arange(10)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    x = lg.arange(10, dtype=np.int32)
    y = np.arange(10, dtype=np.int32)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    x = lg.arange(2.0, 10.0)
    y = np.arange(2.0, 10.0)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    x = lg.arange(2, 30, 3)
    y = np.arange(2, 30, 3)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype

    # xfls = lg.full_like(x, '3', dtype=np.str_)
    # yfls = np.full_like(y, '3', dtype=np.str_)
    # assert(lg.array_equal(xfls, yfls))
    # assert(xfls.dtype == yfls.dtype)

    return


if __name__ == "__main__":
    test()
