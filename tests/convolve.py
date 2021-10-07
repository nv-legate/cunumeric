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
import scipy.signal as sig

import legate.numpy as lg


def test_1d():
    a = lg.random.rand(100)
    v = lg.random.rand(5)

    anp = a.__array__()
    vnp = v.__array__()

    out = lg.convolve(a, v, mode="same")
    out_np = np.convolve(anp, vnp, mode="same")

    assert lg.allclose(out, out_np)


def test_2d():
    a = lg.random.rand(10, 10)
    v = lg.random.rand(3, 5)

    anp = a.__array__()
    vnp = v.__array__()

    out = lg.convolve(a, v, mode="same")
    out_np = sig.convolve(anp, vnp, mode="same")

    assert lg.allclose(out, out_np)


def test_3d():
    a = lg.random.rand(10, 10, 10)
    v = lg.random.rand(3, 5, 3)

    anp = a.__array__()
    vnp = v.__array__()

    out = lg.convolve(a, v, mode="same")
    out_np = sig.convolve(anp, vnp, mode="same")

    assert lg.allclose(out, out_np)


if __name__ == "__main__":
    # test_1d()
    test_2d()
    # test_3d()
