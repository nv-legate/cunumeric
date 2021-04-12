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

from __future__ import division

import numpy as np

import legate.numpy as lg


def test():
    xn = np.array([[1, 2, 3], [4, 5, 6]])
    x = lg.array(xn)
    # print(np.sin(xn))
    # print(lg.sin(x))
    assert np.allclose(np.sin(xn), lg.sin(x))

    # print(np.cos(xn))
    # print(lg.cos(x))
    assert np.allclose(np.cos(xn), lg.cos(x))

    # print(np.sqrt(xn))
    # print(lg.sqrt(x))
    assert np.allclose(np.sqrt(xn), lg.sqrt(x))

    # print(np.exp(xn))
    # print(lg.exp(x))
    assert np.allclose(np.exp(xn), lg.exp(x))

    # print(np.log(xn))
    # print(lg.log(x))
    assert np.allclose(np.log(xn), lg.log(x))

    # print(np.absolute(xn))
    # print(lg.absolute(x))
    assert np.allclose(np.absolute(xn), lg.absolute(x))

    y = lg.tanh(x)
    yn = np.tanh(xn)
    assert np.allclose(y, yn)

    y = lg.cos(0.5)
    # print(y)
    assert np.allclose(y, np.cos(0.5))

    y = lg.sqrt(0.5)
    # print(y)
    assert np.allclose(y, np.sqrt(0.5))

    y = lg.sin(0.5)
    # print(y)
    assert np.allclose(y, np.sin(0.5))

    y = lg.exp(2)
    # print(y)
    assert np.allclose(y, np.exp(2))

    y = lg.log(2)
    # print(y)
    assert np.allclose(y, np.log(2))

    y = lg.absolute(-3)
    # print(y)
    assert y == 3

    np.random.seed(42)
    an = np.random.randn(1, 3, 16)
    bn = 1.0 / (1.0 + np.exp(-an[0, :, :]))
    a = lg.array(an)
    b = 1.0 / (1.0 + lg.exp(-a[0, :, :]))
    assert np.allclose(b, bn)

    return


if __name__ == "__main__":
    test()
