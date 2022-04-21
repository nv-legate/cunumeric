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


def test_1d():
    x = num.array([1, 2, 3])
    y = num.array([4, 5, 6])
    z = num.vstack((x, y))

    xnp = np.array([1, 2, 3])
    ynp = np.array([4, 5, 6])
    znp = np.vstack((xnp, ynp))

    assert np.array_equal(z, znp)


def test_2d():
    x = num.array([[1, 2, 3], [4, 5, 6]])
    y = num.array([[7, 8, 9], [10, 11, 12]])
    z = num.vstack((x, y))

    xnp = np.array([[1, 2, 3], [4, 5, 6]])
    ynp = np.array([[7, 8, 9], [10, 11, 12]])
    znp = np.vstack((xnp, ynp))

    assert np.array_equal(z, znp)


def test_3d():
    x = num.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y = num.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
    z = num.vstack((x, y))

    xnp = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    ynp = np.array(
        [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]
    )
    znp = np.vstack((xnp, ynp))

    assert np.array_equal(z, znp)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
