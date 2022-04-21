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

xn = np.array([[1, 2, 3], [4, 5, 6]])


def test_sin():
    x = num.array(xn)
    assert np.allclose(np.sin(xn), num.sin(x))

    for xx in [-10 - 1, -0.5, 0, 0.5, 1, 10]:
        assert np.allclose(num.sin(xx), np.sin(xx))


def test_cos():
    x = num.array(xn)
    assert np.allclose(np.cos(xn), num.cos(x))

    for xx in [-10 - 1, -0.5, 0, 0.5, 1, 10]:
        assert np.allclose(num.cos(xx), np.cos(xx))


def test_sqrt():
    x = num.array(xn)
    assert np.allclose(np.sqrt(xn), num.sqrt(x))

    for xx in [0, 0.5, 1, 10]:
        assert np.allclose(num.sqrt(xx), np.sqrt(xx))


def test_exp():
    x = num.array(xn)
    assert np.allclose(np.exp(xn), num.exp(x))

    for xx in [-10 - 1, -0.5, 0, 0.5, 1, 10]:
        assert np.allclose(num.exp(xx), np.exp(xx))


def test_log():
    x = num.array(xn)
    assert np.allclose(np.log(xn), num.log(x))

    for xx in [0.5, 1, 2, 10]:
        assert np.allclose(num.log(xx), np.log(xx))


def test_absolute():
    x = num.array(xn)
    assert np.allclose(np.absolute(xn), num.absolute(x))

    for xx in [-3, 0, 3]:
        assert num.absolute(xx) == abs(xx)


def test_tanh():
    x = num.array(xn)
    y = num.tanh(x)
    yn = np.tanh(xn)
    assert np.allclose(y, yn)


def test_combination():
    np.random.seed(42)
    an = np.random.randn(1, 3, 16)
    bn = 1.0 / (1.0 + np.exp(-an[0, :, :]))
    a = num.array(an)
    b = 1.0 / (1.0 + num.exp(-a[0, :, :]))
    assert np.allclose(b, bn)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
