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


def test1():
    anp = np.arange(100).reshape(10, 10)
    a = num.arange(100).reshape((10, 10))
    assert np.array_equal(anp, a)

    for shape in [
        (10, 5, 2),
        (5, 2, 10),
        (5, 2, 5, 2),
        (10, 10, 1),
        (10, 1, 10),
        (1, 10, 10),
    ]:
        bnp = np.reshape(anp, shape)
        b = num.reshape(a, shape)
        assert np.array_equal(bnp, b)

    cnp = np.reshape(anp, (100,))
    c = num.reshape(a, (100,))
    assert np.array_equal(cnp, c)

    dnp = np.ravel(anp)
    d = num.ravel(a)
    assert np.array_equal(dnp, d)


def test2():
    anp = np.random.rand(5, 4, 10)
    a = num.array(anp)

    for shape in [
        (10, 2, 10),
        (20, 10),
        (5, 40),
        (200, 1),
        (1, 200),
        (10, 20),
    ]:
        bnp = np.reshape(anp, shape)
        b = num.reshape(a, shape)
        assert np.array_equal(bnp, b)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
