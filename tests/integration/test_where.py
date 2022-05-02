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

np.random.seed(42)

CONDITIONS = [
    [[True, False], [True, True]],
    [[True, False]],
    [True, False],
    False,
]


def test_basic():
    anp = np.array([1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0])
    a = num.array(anp)
    assert num.array_equal(np.where(anp), num.where(a))


@pytest.mark.parametrize("cond", CONDITIONS, ids=str)
def test_condition(cond):
    anp = np.array(cond)
    xnp = np.array([[1, 2], [3, 4]])
    ynp = np.array([[9, 8], [7, 6]])
    a = num.array(anp)
    x = num.array(xnp)
    y = num.array(ynp)
    assert np.array_equal(np.where(anp, xnp, ynp), num.where(a, x, y))


@pytest.mark.skip
def test_extract():
    cnp = np.array(
        [1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0, 1]
    ).reshape(
        (6, 3)
    )  # noqa E501
    c = num.array(cnp)
    bnp = np.random.randn(6, 3)
    b = num.array(bnp)
    assert num.array_equal(num.extract(c, b), np.extract(cnp, bnp))


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
