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


def test_sum():
    anp = np.array([[1, 2, 3], [4, 5, 6]])
    a = num.array(anp)
    r = a.sum(0)
    assert np.array_equal(r, [5, 7, 9])

    r = a.sum(1)
    assert np.array_equal(r, [6, 15])


def test_random():
    bnp = np.random.random((2, 3))
    b = num.array(bnp)
    assert np.allclose(num.sum(b), np.sum(bnp))


def test_randn():
    af = np.random.randn(4, 5)
    bf = num.array(af)
    assert np.allclose(af.mean(0), bf.mean(0))
    assert np.allclose(af.mean(), bf.mean())


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
