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


def test():
    x = num.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    assert np.array_equal(x[0, :], [1, 2, 3, 4])
    assert np.array_equal(x[1:2, :], [[5, 6, 7, 8]])

    assert np.array_equal(x[:, 0], [1, 5, 9, 13])
    assert np.array_equal(x[:, 1], [2, 6, 10, 14])
    assert np.array_equal(x[:, 2], [3, 7, 11, 15])
    assert np.array_equal(x[:, 3], [4, 8, 12, 16])

    x = num.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    y = x[1:4, 1:3]
    xnp = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    ynp = xnp[1:4, 1:3]
    assert np.array_equal(y[0, :], ynp[0, :])
    assert np.array_equal(y[1, :], ynp[1, :])
    assert np.array_equal(y[2, :], ynp[2, :])

    # TODO: Legate needs partition by phase for this to work efficiently
    # z = x[1:4:2, 1:4:2]
    # assert(num.array_equal(z[0, :], [6, 8]))
    # assert(num.array_equal(z[1, :], [14, 16]))

    dnp = np.random.random((2, 3, 4))
    fnp = np.random.random((3, 2))
    d = num.array(dnp)
    f = num.array(fnp)
    d[1, :, 0] = 1
    d[1, :, 1:3] = f
    dnp[1, :, 0] = 1
    dnp[1, :, 1:3] = fnp
    assert np.array_equal(d, dnp)
    print(fnp)
    print(dnp)
    print(d)

    natest = np.random.random((2, 3, 4))
    natestg = num.array(natest)

    firstslice = natestg[0]
    firstslicegold = natest[0]
    assert np.array_equal(firstslice, firstslicegold)

    # TODO: Legate needs 4-D arrays for this to work correctly
    # secondslice = natestg[:,np.newaxis,:,:]
    # secondslicegold = natest[:,np.newaxis,:,:]
    # assert(num.array_equal(secondslice, secondslicegold))

    # TODO: Legate needs 4-D arrays for this to work correctly
    # thirdslice = natestg[np.newaxis]
    # thirdslicegold = natest[np.newaxis]
    # print(thirdslice)
    # print(thirdslicegold)
    # assert(num.array_equal(thirdslice, thirdslicegold))


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
