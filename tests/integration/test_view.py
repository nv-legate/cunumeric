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


def test_basic():
    num.random.seed(10)
    origVals = num.random.randn(2, 3, 4)
    sliceUpdate = num.random.randn(3, 15)
    sliceView = origVals[0]
    origVals[0, :] += sliceUpdate[:, 11:]
    assert num.array_equal(origVals[0, 0, :], sliceView[0, :])

    sliceView[1, :] = num.random.randn(4)
    assert num.array_equal(origVals[0], sliceView)

    xnp = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    ynp = xnp[2:, 2:]
    x = num.array(xnp)
    y = x[2:, 2:]
    assert np.array_equal(ynp, y)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
