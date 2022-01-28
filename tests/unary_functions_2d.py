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
    xn = np.array([[1, 2, 3], [4, 5, 6]])
    x = num.array(xn)
    # print(np.sin(xn))
    # print(num.sin(x))
    assert np.allclose(np.sin(xn), num.sin(x))

    # print(np.cos(xn))
    # print(num.cos(x))
    assert np.allclose(np.cos(xn), num.cos(x))

    # print(np.sqrt(xn))
    # print(num.sqrt(x))
    assert np.allclose(np.sqrt(xn), num.sqrt(x))

    # print(np.exp(xn))
    # print(num.exp(x))
    assert np.allclose(np.exp(xn), num.exp(x))

    # print(np.log(xn))
    # print(num.log(x))
    assert np.allclose(np.log(xn), num.log(x))

    # print(np.absolute(xn))
    # print(num.absolute(x))
    assert np.allclose(np.absolute(xn), num.absolute(x))

    y = num.tanh(x)
    yn = np.tanh(xn)
    assert np.allclose(y, yn)

    y = num.cos(0.5)
    # print(y)
    assert np.allclose(y, np.cos(0.5))

    y = num.sqrt(0.5)
    # print(y)
    assert np.allclose(y, np.sqrt(0.5))

    y = num.sin(0.5)
    # print(y)
    assert np.allclose(y, np.sin(0.5))

    y = num.exp(2)
    # print(y)
    assert np.allclose(y, np.exp(2))

    y = num.log(2)
    # print(y)
    assert np.allclose(y, np.log(2))

    y = num.absolute(-3)
    # print(y)
    assert y == 3

    np.random.seed(42)
    an = np.random.randn(1, 3, 16)
    bn = 1.0 / (1.0 + np.exp(-an[0, :, :]))
    a = num.array(an)
    b = 1.0 / (1.0 + num.exp(-a[0, :, :]))
    assert np.allclose(b, bn)

    return


if __name__ == "__main__":
    test()
